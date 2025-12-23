#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
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
from muna.beta.openai.annotations import Annotations
from numpy import array, int32, ndarray, ones_like
from onnxruntime import InferenceSession
from re import findall
from requests import get
from torch import inference_mode, Tensor
from torch.hub import load_state_dict_from_url
from typing import Annotated

# Load NeuTTS Air
model_path = hf_hub_download(
    repo_id="neuphonic/neutts-air-q4-gguf",
    filename="neutts-air-Q4-0.gguf"
)
llama_model = Llama(
    model_path=model_path,
    verbose=False,
    n_gpu_layers=0,  # CPU only (would be -1 for GPU)
    n_ctx=32768,     # Use model's training context size
    mlock=True,
    logits_all=True, # Enable logits for all tokens
    # flash_attn=False  # They set this to False for CPU (line 105)
)

# Load ByT5 ONNX model for phonemization (avoids tokenizer serialization issues)
byt5_model_path = hf_hub_download(
    "OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx",
    "fdemelo_g2p-mbyt5-12l-ipa-childes-espeak.onnx"
)
byt5_model = InferenceSession(byt5_model_path)

# Download ONNX decoder model
onnx_decoder_path = hf_hub_download(
    repo_id="neuphonic/neucodec-onnx-decoder",
    filename="model.onnx"
)

# Create ONNX Runtime session for decoder
neucodec_session = InferenceSession(onnx_decoder_path)

def _load_neutts_voice(voice_name: str) -> tuple[ndarray, str]:
    """Load reference codes and text for a single voice"""
    reference_codes_tensor = load_state_dict_from_url(f"https://raw.githubusercontent.com/neuphonic/neutts-air/main/samples/{voice_name}.pt")
    reference_text = get(f"https://raw.githubusercontent.com/neuphonic/neutts-air/main/samples/{voice_name}.txt").text.strip()
    # Convert PyTorch tensor to numpy array
    reference_codes = reference_codes_tensor.numpy()
    return reference_codes, reference_text

# Load all available voices into a single resource
AVAILABLE_VOICES = ["dave", "jo"]

# Create voice references dictionary with numpy arrays and text
voice_data = {}
for voice_name in AVAILABLE_VOICES:
    codes, text = _load_neutts_voice(voice_name)
    voice_data[f"{voice_name}_codes"] = codes
    voice_data[f"{voice_name}_text"] = text

# Save to NPZ file (following Kokoro pattern)
from numpy import savez
neutts_voices_path = "neutts_voices.npz"
savez(neutts_voices_path, **voice_data)

# Load voices from NPZ file
from numpy import load
voices_npz = load(neutts_voices_path)

# Reconstruct voice references from loaded data
VOICE_REFERENCES = {
    voice_name: (voices_npz[f"{voice_name}_codes"], str(voices_npz[f"{voice_name}_text"]))
    for voice_name in AVAILABLE_VOICES
}

@compile(
    tag="@neuphonic/neutts-air",
    description="Perform text-to-speech with NeuTTS-Air.",
    access="public",
    sandbox=Sandbox()
        .pip_install("torch", "torchaudio", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "llama-cpp-python", "onnxruntime", "transformers"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=neucodec_session, model_path=onnx_decoder_path),
        OnnxRuntimeInferenceSessionMetadata(session=byt5_model, model_path=byt5_model_path),
        LlamaCppInferenceMetadata(
            model=llama_model,
        )
    ]
)
@inference_mode()
def generate_speech(
    text: Annotated[str, Parameter.Generic(description="Text to generate speech from.")],
    *,
    language: Annotated[str, Parameter.Generic(description="Generation language.")]="en-US",
    voice: Annotated[str, Annotations.AudioVoice(description="Generation voice.")],
) -> Annotated[ndarray, Parameter.Audio(
    description="Linear PCM audio samples with shape (F,) and sample rate 24KHz.",
    sample_rate=24_000
)]:
    """
    Perform text-to-speech with NeuTTS-Air.
    """
    # Get voice-specific reference codes and text
    if voice not in VOICE_REFERENCES:
        raise ValueError(f"Unsupported voice: {voice}. Available voices: {list(VOICE_REFERENCES.keys())}")
    reference_codes, reference_text = VOICE_REFERENCES[voice]
    # Create prompt with IPA phonemes
    prompt = _create_prompt(text, reference_codes, reference_text, language)
    # Generate tokens using direct model call (no chat template)
    # Use more tokens for jo voice since it has longer prompts
    max_tokens = 4096 if voice == "jo" else 2048
    print(f"Using max_tokens: {max_tokens}")
    # Use official NeuTTS parameters from their _infer_ggml method
    response = llama_model(
        prompt,
        max_tokens=2048,  # Use their max_context value
        temperature=1.0,
        top_k=50,
        echo=False,  # Don't echo the prompt
        stop=["<|SPEECH_GENERATION_END|>"],  # Stop at end token
    )
    # Extract generated text
    generated_text = response["choices"][0]["text"]
    finish_reason = response["choices"][0].get("finish_reason", "unknown")
    usage = response.get("usage", { })
    # Check if stop token is in the text
    stop_token = "<|SPEECH_GENERATION_END|>"
    if stop_token in generated_text:
        stopped_generated_text = generated_text.split(stop_token)[0]
        print("Found stop token, truncated generation")
    else:
        stopped_generated_text = generated_text
        print("No stop token found")
    # Parse speech tokens from the generated text in format <|speech_XXXXX|>
    # Extract all <|speech_XXXXX|> tokens using regex (exactly like official _decode method line 228)
    # IMPORTANT: Only extract tokens from the generated text, NOT from the prompt
    pattern = r"<\|speech_(\d+)\|>"
    matches = findall(pattern, stopped_generated_text)
    speech_tokens = [int(match) for match in matches]
    if len(speech_tokens) < 1:
        raise ValueError("No valid speech tokens found in the output. Generated text: " + generated_text[:200])
    # Decode speech tokens to audio
    audio_array = _decode_speech_tokens(speech_tokens)
    return audio_array

def _convert_to_ipa(
    text: str,
    *,
    language: str
) -> str:
    """
    Convert a string to phonemes using ONNX ByT5 model with word-by-word phonemization.
    This approach produces more accurate results than full-text phonemization.
    Preserves sentence boundary punctuation (periods, question marks, exclamation marks) for prosody.
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
            logits = byt5_model.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": array(decoder_input_ids)[None]
            })[0]
            last_token_logits = logits[:,-1,:]
            next_id = int(last_token_logits.argmax(-1).item())
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

def _decode_speech_tokens(speech_tokens: list[int]) -> ndarray:
    """
    Decode speech tokens using ONNX model.
    """
    if len(speech_tokens) < 1:
        # Return silence if no tokens
        return array([])
    # Convert to numpy array and reshape for ONNX model
    codes = array(speech_tokens, dtype=int32)[None, None, :]
    # Run ONNX inference directly
    outputs = neucodec_session.run(None, { "codes": codes })
    # Return flattened audio
    return outputs[0][0, 0, :]

def _create_prompt(
    input_text: str,
    reference_codes: Tensor,
    reference_text: str,
    language: str
) -> str:
    """
    Create input prompt by phonemizing input text.
    """
    # Convert to phonemes exactly like their implementation (lines 307-308)
    ref_phonemes = _convert_to_ipa(reference_text, language=language)
    input_phonemes = _convert_to_ipa(input_text, language=language)
    # Create reference codes string exactly like line 310
    # reference_codes is always a numpy array from _encode_reference_audio, so we can iterate directly
    codes_str = "".join([f"<|speech_{idx.item()}|>" for idx in reference_codes])
    # Use their exact prompt format from lines 311-314
    prompt_part1 = f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_phonemes} {input_phonemes}<|TEXT_PROMPT_END|>"
    prompt_part2 = f"assistant:<|SPEECH_GENERATION_START|>{codes_str}"
    # prompt = prompt_part1 + "\n" + prompt_part2
    prompt = prompt_part1 + prompt_part2
    return prompt

if __name__ == "__main__":
    import sounddevice as sd
    prompt_text = "It was the best of times. It was the worst of times."
    voice = "jo"
    print(f"Testing: '{prompt_text}' with {voice} voice")
    audio = generate_speech(prompt_text, voice=voice)
    sd.play(audio, samplerate=24_000)
    sd.wait()
