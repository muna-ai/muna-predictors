#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["librosa", "muna", "torch", "transformers"]
# ///

from copy import deepcopy
from muna import compile, BatchConfig, Parameter, Sandbox
from muna.beta import KVCacheConfig, TensorRTInferenceMetadata
from numpy import ndarray
import torch
from torch import from_numpy, hann_window, inference_mode, Tensor
from torch.export import Dim
from torch.nn import Module
from transformers import (
    AutoTokenizer, WhisperFeatureExtractor,
    WhisperForConditionalGeneration
)
from transformers.cache_utils import EncoderDecoderCache
from typing import Annotated

# Load PyTorch model
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    low_cpu_mem_usage=True
)
model.eval()
_sample_rate = 16_000
_num_layers = model.config.decoder_layers
_batch_dim = Dim("batch", min=1, max=16)

class WhisperEncoder(Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = deepcopy(encoder).half()
    def forward(self, mel):
        return self.encoder(mel.half()).last_hidden_state.float()

class WhisperDecoderPrefill(Module):
    def __init__(self, decoder, proj_out, num_layers):
        super().__init__()
        self.decoder = deepcopy(decoder).half()
        self.proj_out = deepcopy(proj_out).half()
        self.num_layers = num_layers
    def forward(self, input_ids, encoder_hidden_states):
        dec_out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states.half(),
            use_cache=True,
        )
        logits = self.proj_out(dec_out.last_hidden_state).float()
        kv = dec_out.past_key_values
        self_keys = torch.stack([kv.self_attention_cache.layers[i].keys for i in range(self.num_layers)])
        self_values = torch.stack([kv.self_attention_cache.layers[i].values for i in range(self.num_layers)])
        cross_keys = torch.stack([kv.cross_attention_cache.layers[i].keys for i in range(self.num_layers)])
        cross_values = torch.stack([kv.cross_attention_cache.layers[i].values for i in range(self.num_layers)])
        return logits, self_keys, self_values, cross_keys, cross_values

class WhisperDecoderDecode(Module):
    def __init__(self, decoder, proj_out, num_layers):
        super().__init__()
        self.decoder = deepcopy(decoder).half()
        self.proj_out = deepcopy(proj_out).half()
        self.num_layers = num_layers
    def forward(self, input_ids, encoder_hidden_states,
                self_keys, self_values, cross_keys, cross_values):
        cache_tuples = []
        for i in range(self.num_layers):
            cache_tuples.append((self_keys[i], self_values[i], cross_keys[i], cross_values[i]))
        past_kv = EncoderDecoderCache(cache_tuples)
        dec_out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states.half(),
            past_key_values=past_kv,
            use_cache=True,
        )
        logits = self.proj_out(dec_out.last_hidden_state).float()
        new_kv = dec_out.past_key_values
        new_self_keys = torch.stack([new_kv.self_attention_cache.layers[i].keys for i in range(self.num_layers)])
        new_self_values = torch.stack([new_kv.self_attention_cache.layers[i].values for i in range(self.num_layers)])
        return logits, new_self_keys, new_self_values

# Load tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3")
_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

# Precompute mel filterbank and STFT window
N_FFT = _feature_extractor.n_fft
HOP_LENGTH = _feature_extractor.hop_length
MEL_FILTERS = from_numpy(_feature_extractor.mel_filters).float()
STFT_WINDOW = hann_window(N_FFT)

_CHUNK_SAMPLES = 480000 # 30s window at 16kHz
_MAX_BATCH = _batch_dim.max

@compile(
    tag="@yusuf/whisper-large-v3-trt-h100",
    targets=["x86_64-unknown-linux-gnu"],
    access="unlisted",
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "transformers"),
    metadata=[
        TensorRTInferenceMetadata(
            model=model,
            kv_cache=KVCacheConfig(
                encoder=WhisperEncoder(model.model.encoder).eval(),
                prefill=WhisperDecoderPrefill(model.model.decoder, model.proj_out, _num_layers).eval(),
                decode=WhisperDecoderDecode(model.model.decoder, model.proj_out, _num_layers).eval(),
            ),
            input_shapes=[(_batch_dim, 128, 3000)],
            cuda_arch="sm_90",              # target Hopper
            hardware_compatibility="none"   # enable Hopper-specific optimizations
        ),
    ]
)
@inference_mode()
def whisper_large_v3(
    audio: Annotated[
        list[ndarray],
        Parameter.Audio(
            sample_rate=_sample_rate,
            description="Audio to transcribe with shape (B,F).",
            batch=BatchConfig(max_count=_batch_dim.max)
        )
    ]
) -> Annotated[
    list[str],
    Parameter.Generic(description="Transcribed text from the audio.")
]:
    """
    Transcribe audio to text with OpenAI Whisper Large V3.
    """
    # Chunk each audio input into 30s segments
    waveforms = [from_numpy(item).flatten() for item in audio]
    chunks = [waveforms[0][:_CHUNK_SAMPLES]]
    chunk_owners = [0]
    chunks.clear()
    chunk_owners.clear()
    for idx, w in enumerate(waveforms):
        for start in range(0, max(1, w.shape[0]), _CHUNK_SAMPLES):
            chunks.append(w[start:start + _CHUNK_SAMPLES])
            chunk_owners.append(idx)
    # Transcribe chunks in batches that fit the TRT engine
    chunk_texts = [""]
    chunk_texts.clear()
    for batch_start in range(0, len(chunks), _MAX_BATCH):
        chunk_texts += _transcribe_chunks(chunks[batch_start:batch_start + _MAX_BATCH])
    # Concatenate chunk transcriptions per original audio input
    results = [""] * len(audio)
    for chunk_idx, audio_idx in enumerate(chunk_owners):
        text = chunk_texts[chunk_idx].strip()
        if text:
            results[audio_idx] = (results[audio_idx] + " " + text).strip()
    # Return
    return results

def _transcribe_chunks(waveforms: list[Tensor]) -> list[str]:
    """
    Transcribe a batch of waveform chunks, each ≤30s.
    """
    mels = [_compute_window_log_mel(w) for w in waveforms]
    mel_batch = torch.stack(mels)
    token_ids = model.generate(
        input_features=mel_batch,
        max_new_tokens=444,
        language="en",
        task="transcribe",
    )
    results = [tokenizer.decode(
        token_ids[i],
        skip_special_tokens=True
    ) for i in range(len(waveforms))]
    return results

def _compute_window_log_mel(waveform: Tensor) -> Tensor:
    """
    Pad or truncate waveform to a 30s window, then compute log mel spectrogram.
    """
    pad_size = max(0, 480000 - waveform.shape[0])
    waveform = torch.nn.functional.pad(waveform[:480000], (0, pad_size))
    stft = torch.stft(waveform, N_FFT, HOP_LENGTH, window=STFT_WINDOW, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = MEL_FILTERS.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

if __name__ == "__main__":
    from json import dumps
    import librosa
    from pathlib import Path
    # Load audio
    audio_path = Path(__file__).parent / "demo" / "speech.wav"
    data, sr = librosa.load(audio_path, sr=_sample_rate, mono=True)
    # Transcribe
    transcriptions = whisper_large_v3([data])
    # Log
    print(dumps(transcriptions, indent=2))