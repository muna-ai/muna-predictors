#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub",
#   "librosa",
#   "llama-cpp-python",
#   "muna",
#   "onnxruntime",
#   "sounddevice",
#   "soundfile",
#   "torch",
#   "transformers"
# ]
# ///

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from muna import compile, Parameter, Sandbox
from muna.beta import LlamaCppInferenceMetadata, OnnxRuntimeInferenceSessionMetadata
from muna.beta.openai import Annotations
from numpy import array, int32, load, ndarray, ones_like, savez
from numpy.lib.npyio import NpzFile
from onnxruntime import InferenceSession
from pathlib import Path
from re import findall
from requests import get
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Annotated, Literal

GenerationVoice = Literal["dave", "jo"]

# Load NeuTTS Air
model_path = hf_hub_download(
    repo_id="neuphonic/neutts-air-q4-gguf",
    filename="neutts-air-Q4-0.gguf"
)
model = Llama(
    model_path=model_path,
    verbose=False,
    n_gpu_layers=0,  # CPU only (would be -1 for GPU)
    n_ctx=32768,     # Use model's training context size
    mlock=True,
    logits_all=True
)

# Load NeuCodec decoder
decoder_path = hf_hub_download(
    repo_id="neuphonic/neucodec-onnx-decoder",
    filename="model.onnx"
)
decoder = InferenceSession(decoder_path)

# Load phonemizer ByT5 ONNX model
phonemizer_path = hf_hub_download(
    repo_id="OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx",
    filename="fdemelo_g2p-mbyt5-12l-ipa-childes-espeak.onnx"
)
phonemizer = InferenceSession(phonemizer_path)

# Load reference voices
voices_path = Path("voices.npz")
voices_base_url = "https://raw.githubusercontent.com/neuphonic/neutts-air/main/samples"
if not voices_path.exists():
    savez(voices_path, **{
        "dave": load_state_dict_from_url(f"{voices_base_url}/dave.pt").numpy(),
        "jo": load_state_dict_from_url(f"{voices_base_url}/jo.pt").numpy(),
    })
voice_codes: NpzFile = load(voices_path)
voice_prompts = {
    "dave": get(f"{voices_base_url}/dave.txt").text.strip(),
    "jo": get(f"{voices_base_url}/jo.txt").text.strip(),
}

@compile(
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "llama-cpp-python", "onnxruntime", "transformers"),
    metadata=[
        LlamaCppInferenceMetadata(
            model=model,
            #backends=["cuda"]
        ),
        OnnxRuntimeInferenceSessionMetadata(session=decoder, model_path=decoder_path),
        OnnxRuntimeInferenceSessionMetadata(session=phonemizer, model_path=phonemizer_path)
    ]
)
def neutts_air(
    text: Annotated[
        str,
        Parameter.Generic(description="Text to generate speech from.")
    ],
    *,
    voice: Annotated[
        GenerationVoice,
        Annotations.AudioVoice(description="Generation voice.")
    ],
    language: Annotated[
        str,
        Parameter.Generic(description="Generation language.")
    ]="en-US"
) -> Annotated[ndarray, Parameter.Audio(
    description="Linear PCM audio samples with shape (F,) and sample rate 24KHz.",
    sample_rate=24_000
)]:
    """
    Perform text-to-speech with NeuTTS-Air.
    """
    # Get voice reference codes and text
    reference_codes = voice_codes[voice]
    reference_text = voice_prompts[voice]
    # Create prompt with IPA phonemes
    prompt = _create_prompt(text, reference_codes, reference_text, language)
    # Run NeuTTS-Air
    response: dict = model(
        prompt,
        max_tokens=2048,
        temperature=1.0,
        top_k=50,
        echo=False,
        stop=["<|SPEECH_GENERATION_END|>"]
    )
    # Extract generated text
    generated_text: str = response["choices"][0]["text"]
    stop_token = "<|SPEECH_GENERATION_END|>"
    if stop_token in generated_text:
        stopped_generated_text = generated_text.split(stop_token)[0]
    else:
        stopped_generated_text = generated_text
    # Parse speech tokens from the generated text in format <|speech_XXXXX|>
    pattern = r"<\|speech_(\d+)\|>"
    matches = findall(pattern, stopped_generated_text)
    speech_tokens = [int(match) for match in matches]
    if len(speech_tokens) < 1:
        raise ValueError(
             "No valid speech tokens found in the output. Generated text:"
             + generated_text[:200]
        )
    # Decode speech tokens to audio
    audio_array = _decode_speech_tokens(speech_tokens)
    # Return
    return audio_array

def _decode_speech_tokens(speech_tokens: list[int]) -> ndarray:
    """
    Decode speech tokens using ONNX model
    """
    # Return silence if no tokens
    if len(speech_tokens) < 1:
        return array([])
    # Convert to numpy array and reshape for ONNX model
    codes = array(speech_tokens, dtype=int32)[None, None, :]
    # Run ONNX inference directly
    outputs = decoder.run(None, { "codes": codes })
    # Return flattened audio
    return outputs[0][0, 0, :]

def _create_prompt(
    input_text: str,
    reference_codes: Tensor,
    reference_text: str,
    language: str
) -> str:
    """
    Create speech input prompt.
    """
    # Convert to phonemes
    ref_phonemes = _convert_to_ipa(reference_text, language=language)
    input_phonemes = _convert_to_ipa(input_text, language=language)
    codes_str = "".join([f"<|speech_{idx.item()}|>" for idx in reference_codes])
    # Construct prompt
    prompt_part1 = "user: Convert the text to speech:<|TEXT_PROMPT_START|>"
    prompt_part2 = f"{ref_phonemes} {input_phonemes}<|TEXT_PROMPT_END|>"
    prompt_part3 = f"assistant:<|SPEECH_GENERATION_START|>{codes_str}"
    prompt = prompt_part1 + prompt_part2 + prompt_part3
    # Return
    return prompt

def _convert_to_ipa(
    text: str,
    *,
    language: str
) -> str:
    """
    Convert a string to phonemes using a ByT5 model with word-by-word phonemization.
    This approach produces more accurate results than full-text phonemization.
    Preserves sentence boundary punctuation (periods, question marks,
    exclamation marks) for prosody.
    """
    # Split text into words while preserving sentence boundary punctuation
    words = text.split()
    word_phonemes = [""]
    for word in words:
        # Create a local copy to modify
        current_word = word
        # Extract sentence boundary punctuation using simple string operations
        sentence_end = ""
        if word.endswith("."):
            sentence_end = "."
            current_word = word[:-1]
        elif word.endswith("!"):
            sentence_end = "!"
            current_word = word[:-1]
        elif word.endswith("?"):
            sentence_end = "?"
            current_word = word[:-1]
        # Remove all punctuation for phonemization (simple character filtering)
        clean_word = ""
        for each_char in current_word:
            if each_char.isalnum():
                clean_word += each_char
        if len(clean_word) == 0:
            # If word is pure punctuation, preserve it if it's sentence-ending
            if len(sentence_end) > 0:
                word_phonemes.append(sentence_end)
            continue
        # Phonemize the clean word using ONNX model
        prefix = "<" + language + ">: "
        prompt = prefix + clean_word
        # num_special_tokens = 3
        encoded_prompt = prompt.encode()
        input_ids = array([list(encoded_prompt)]) + 3  # num_special_tokens
        attention_mask = ones_like(input_ids)
        # Generate autoregressively
        eos_token_id = 1
        max_length = 200
        decoder_input_ids = [0]
        for idx in range(max_length):
            logits = phonemizer.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": array(decoder_input_ids)[None]
            })[0]
            last_token_logits = logits[:,-1,:]
            next_id = int(last_token_logits.argmax(axis=-1).item())
            decoder_input_ids.append(next_id)
            if next_id == eos_token_id:
                break
        # Decode to IPA
        output_ids = [token_id - 3 for token_id in decoder_input_ids]
        phoneme_data = bytes([token_id for token_id in output_ids if 0 <= token_id < 256])
        phonemes = phoneme_data.decode()
        # Attach sentence boundary punctuation directly to the phonemes (no space)
        if len(sentence_end) > 0:
            phonemes = phonemes + sentence_end
        word_phonemes.append(phonemes)
    trimmed_word_phonemes = word_phonemes[1:]
    # Join phonemes with spaces
    return " ".join(trimmed_word_phonemes)

if __name__ == "__main__":
    import sounddevice as sd
    # Generate audio
    audio = neutts_air(
        "It was the best of times. It was the worst of times.",
        voice="dave"
    )
    # Playback
    sd.play(audio, samplerate=24_000)
    sd.wait()