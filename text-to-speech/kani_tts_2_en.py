#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["kani-tts-2", "muna", "transformers==4.56.0"]
# ///

from kani_tts.model import KaniTTS2ForCausalLM
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, OnnxRuntimeInferenceMetadata
from nemo.collections.tts.models import AudioCodecModel
import numpy as np
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedTextConfig
from transformers import AutoTokenizer
from typing import Annotated, Literal
import torch
import torch.nn as nn

GenerationAccent = Literal[
    "en_us", "en_nyork", "en_oakl",
    "en_glasg", "en_bost", "en_scou"
]

# Token configuration
TEXT_VOCAB_SIZE = 64400
END_OF_TEXT = 2
START_OF_SPEECH = TEXT_VOCAB_SIZE + 1    # 64401
END_OF_SPEECH = TEXT_VOCAB_SIZE + 2      # 64402
START_OF_HUMAN = TEXT_VOCAB_SIZE + 3     # 64403
END_OF_HUMAN = TEXT_VOCAB_SIZE + 4       # 64404
AUDIO_TOKENS_START = TEXT_VOCAB_SIZE + 10 # 64410
CODEBOOK_SIZE = 4032
SAMPLE_RATE = 22050

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("nineninesix/kani-tts-2-en")

# Load KaniTTS2 model
model = KaniTTS2ForCausalLM.from_pretrained(
    "nineninesix/kani-tts-2-en",
    audio_tokens_start=AUDIO_TOKENS_START,
    torch_dtype=torch.float32,
    device_map="cpu",
    attn_implementation="eager",
).eval()

# Create a wrapper module that runs dequantize + decode
class NanoCodecDecoderWrapper(nn.Module):

    def __init__(self, codec_model):
        super().__init__()
        self.vector_quantizer = codec_model.vector_quantizer
        self.audio_decoder = codec_model.audio_decoder

    def forward(self, audio_codes, tokens_len):
        # NanoCodec vector_quantizer.decode expects (C, B, T) but we have (B, C, T)
        codes_cbt = audio_codes.permute(1, 0, 2)
        dequantized = self.vector_quantizer.decode(indices=codes_cbt, input_len=tokens_len)
        audio, audio_len = self.audio_decoder(inputs=dequantized, input_len=tokens_len)
        return audio

# Load codec
raw_codec = AudioCodecModel.from_pretrained("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")
codec = NanoCodecDecoderWrapper(raw_codec).eval().cpu()

class KaniModelOnnxConfig(OnnxConfig):
    DEFAULT_ONNX_OPSET = 18
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self):
        return { "input_ids": { 0: "batch", 1: "seq_len" } }

    @property
    def outputs(self):
        return { "logits": { 0: "batch", 1: "seq_len" } }

    def generate_dummy_inputs(self, framework="pt", **kwargs):
        inputs = { "input_ids": torch.randint(0, 80000, (1, 50), dtype=torch.long) }
        inputs = { k: v.numpy() if framework == "np" else v for k, v in inputs.items() }
        return inputs

@compile(
    sandbox=Sandbox()
        .pip_install("torch", "torchaudio", "torchcodec", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("kani-tts-2")
        .pip_install("transformers==4.56.0", "optimum[onnx]"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            exporter="optimum",
            optimum_config=KaniModelOnnxConfig(model.config, task="text-generation"),
            providers=["cpu"]
        ),
        OnnxRuntimeInferenceMetadata(
            model=codec,
            model_args=[
                torch.randint(0, CODEBOOK_SIZE, (1, 4, 50), dtype=torch.long),
                torch.tensor([50], dtype=torch.long)
            ],
            input_shapes=[(1, 4, "frames"), ("frames",)],
            exporter="torchscript",
            providers=["cpu"]
        ),
    ]
)
@torch.inference_mode()
def kani_tts_2_en(
    text: Annotated[
        str,
        Parameter.Generic(description="Text to synthesize to speech.")
    ],
    *,
    accent: Annotated[
        GenerationAccent,
        Annotations.AudioVoice(description="English accent variant.")
    ]
) -> Annotated[np.ndarray, Parameter.Audio(
    sample_rate=SAMPLE_RATE,
    description="Generated speech audio at 22kHz sample rate."
)]:
    """
    Generate speech from text using Kani TTS 2 (english).
    """
    # Build input IDs with special tokens: [START_OF_HUMAN, text_ids, END_OF_TEXT, END_OF_HUMAN]
    input_ids = tokenizer(f"{accent}: {text}", return_tensors="pt")["input_ids"]
    start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
    # Generate audio tokens
    output_ids = model.generate(
        modified_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3000,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=END_OF_SPEECH,
    )
    # Extract audio codes from generated tokens
    out = output_ids[0]
    audio = _extract_and_decode_audio(out)
    # Return
    return audio

def _extract_and_decode_audio(out_ids: torch.Tensor) -> np.ndarray:
    """
    Extract audio codes from token sequence and decode to audio waveform.
    """
    # Find speech boundary markers
    start_idx = (out_ids == START_OF_SPEECH).nonzero(as_tuple=True)[0].item()
    end_idx = (out_ids == END_OF_SPEECH).nonzero(as_tuple=True)[0].item()
    audio_tokens = out_ids[start_idx + 1 : end_idx]
    # Ensure we have complete frames (4 tokens per frame)
    num_frames = len(audio_tokens) // 4
    if num_frames == 0:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)
    audio_tokens = audio_tokens[:num_frames * 4].reshape(-1, 4)
    # Convert token IDs to codec codes: subtract codebook offsets and audio token start
    offsets = torch.tensor([CODEBOOK_SIZE * i for i in range(4)])
    audio_codes = audio_tokens - offsets - AUDIO_TOKENS_START
    # Decode with PyTorch NanoCodec: shape (1, 4, num_frames)
    audio_codes_t = audio_codes.T.unsqueeze(0).to(torch.long)
    tokens_len = torch.tensor([num_frames], dtype=torch.long)
    audio_output = codec(audio_codes_t, tokens_len)
    # Return
    return audio_output.squeeze().numpy()

if __name__ == "__main__":
    import sounddevice as sd
    # Create speech
    audio = kani_tts_2_en(
        "Hello, this is a test of the Kani TTS 2 text to speech model.",
        accent="en_oakl"
    )
    # Play audio
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()