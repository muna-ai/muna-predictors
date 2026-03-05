#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "librosa", "muna", "onnxruntime", "transformers"]
# ///

from huggingface_hub import hf_hub_download
from json import loads
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from numpy import argmax, array, bool_, float32, int64, ndarray, zeros
from onnxruntime import InferenceSession
from pathlib import Path
from transformers import AutoTokenizer
from typing import Annotated

REPO_ID = "UsefulSensors/moonshine"
MODEL_DIR = "onnx/merged/base/float"
MAX_GENERATE_LENGTH = 512

# Download models
encoder_model_path = hf_hub_download(REPO_ID, f"{MODEL_DIR}/encoder_model.onnx")
decoder_model_path = hf_hub_download(REPO_ID, f"{MODEL_DIR}/decoder_model_merged.onnx")
config_path = hf_hub_download(REPO_ID, f"{MODEL_DIR}/config.json")
preprocessor_config_path = hf_hub_download(REPO_ID, f"{MODEL_DIR}/preprocessor_config.json")
generation_config_path = hf_hub_download(REPO_ID, f"{MODEL_DIR}/generation_config.json")

# Load model and tokenizer
encoder_session = InferenceSession(encoder_model_path)
decoder_session = InferenceSession(decoder_model_path)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder=MODEL_DIR)

# Load config
config = loads(Path(config_path).read_text())
preprocessor_config = loads(Path(preprocessor_config_path).read_text())
generation_config = loads(Path(generation_config_path).read_text())
sample_rate: int = preprocessor_config["sampling_rate"]
decoder_start_token_id: int = generation_config["decoder_start_token_id"]
eos_token_id: int = generation_config["eos_token_id"]
num_kv_heads: int = config["decoder_num_key_value_heads"]
dim_kv: int = config["hidden_size"] // config["decoder_num_attention_heads"]

_kv_cache_names: list[str] = [
    inp.name for inp in decoder_session.get_inputs()
    if "past_key_values" in inp.name
]

@compile(
    sandbox=Sandbox()
        .pip_install("huggingface_hub", "onnxruntime", "transformers"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(
            session=encoder_session,
            model_path=encoder_model_path
        ),
        OnnxRuntimeInferenceSessionMetadata(
            session=decoder_session,
            model_path=decoder_model_path,
            providers=["cpu", "cuda"]
        ),
    ],
)
def moonshine_base(
    audio: Annotated[ndarray, Parameter.Audio(
        sample_rate=sample_rate,
        description="Audio to transcribe with shape (F,C)."
    )],
) -> Annotated[
    str,
    Parameter.Generic(description="Transcribed text from the audio.")
]:
    """
    Transcribe audio to text with Moonshine (base).
    """
    samples = audio.astype(float32).flatten()
    input_values = samples.reshape(1, -1)
    encoder_outputs = encoder_session.run(None, { "input_values": input_values })[0]
    input_ids = array([[decoder_start_token_id]], dtype=int64)
    past_key_values = { "": zeros([1, num_kv_heads, 0, dim_kv], dtype=float32) }
    past_key_values.clear()
    for name in _kv_cache_names:
        past_key_values[name] = zeros([1, num_kv_heads, 0, dim_kv], dtype=float32)
    generated_ids: list[int] = [0]
    for i in range(MAX_GENERATE_LENGTH):
        use_cache_branch = i > 0
        decoder_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_outputs,
            "use_cache_branch": array([use_cache_branch], dtype=bool_),
        }
        decoder_inputs.update(past_key_values)
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]
        present_key_values = decoder_outputs[1:]
        next_token_id = int(argmax(logits[:, -1, :], axis=-1).item())
        if next_token_id == eos_token_id:
            break
        generated_ids.append(next_token_id)
        input_ids = array([[next_token_id]], dtype=int64)
        for j, name in enumerate(_kv_cache_names):
            if not use_cache_branch or "decoder" in name:
                past_key_values[name] = present_key_values[j]
    # Decode
    return tokenizer.decode(generated_ids[1:], skip_special_tokens=True)

if __name__ == "__main__":
    import librosa
    data, _ = librosa.load("test/media/librispeech_sample.wav", sr=sample_rate, mono=True)
    text = moonshine_base(data)
    print(text)