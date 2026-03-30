#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub",
#   "llama-cpp-python",
#   "muna",
#   "onnxruntime",
#   "sounddevice",
#   "torch",
#   "transformers"
# ]
# ///

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from muna import compile, Parameter, Sandbox
from muna.beta import (
    Annotations, LlamaCppInferenceMetadata,
    OnnxRuntimeInferenceSessionMetadata
)
from numpy import array, int32, load, ndarray, ones_like, savez
from numpy.lib.npyio import NpzFile
from onnxruntime import InferenceSession
from pathlib import Path
from re import findall
from requests import get
from torch.hub import load_state_dict_from_url
from typing import Annotated, Literal

GenerationVoice = Literal[
    "dave",
    #"jo"
]

# Load NeuTTS Nano (Q8 GGUF)
model_path = hf_hub_download(
    repo_id="neuphonic/neutts-nano-q8-gguf",
    filename="neutts-nano-Q8_0.gguf"
)
model = Llama(
    model_path=model_path,
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=2048,
    mlock=True,
    logits_all=True
)

# Load NeuCodec ONNX decoder
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
voices_path = Path("neutts_nano_voices.npz")
voices_base_url = "https://raw.githubusercontent.com/neuphonic/neutts/main/samples"
if not voices_path.exists():
    savez(voices_path, **{
        "dave": load_state_dict_from_url(f"{voices_base_url}/dave.pt").numpy(),
        "jo": load_state_dict_from_url(f"{voices_base_url}/jo.pt").numpy(),
    })
voice_codes: NpzFile = load(voices_path)
voice_prompts = {
    "dave": get(f"{voices_base_url}/dave.txt").text.strip(),
    "jo": "So I just got back from the gym. I've been trying this new routine that my friend recommended. It's honestly been a game changer for my energy levels throughout the day.",
}

@compile(
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("huggingface_hub", "llama-cpp-python", "onnxruntime", "transformers"),
    metadata=[
        LlamaCppInferenceMetadata(model=model),
        OnnxRuntimeInferenceSessionMetadata(
            session=decoder,
            model_path=decoder_path,
            providers=["cpu", "cuda"]
        ),
        OnnxRuntimeInferenceSessionMetadata(
            session=phonemizer,
            model_path=phonemizer_path,
            providers=["cpu", "cuda"]
        )
    ]
)
def neutts_nano(
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
    Perform text-to-speech with NeuTTS Nano.
    """
    reference_codes = voice_codes[voice]
    reference_text = voice_prompts[voice]
    prompt = _create_prompt(text, reference_codes, reference_text, language)
    response: dict = model(
        prompt,
        max_tokens=2048,
        temperature=1.0,
        top_k=50,
        echo=False,
        stop=["<|SPEECH_GENERATION_END|>"]
    )
    generated_text: str = response["choices"][0]["text"]
    stop_token = "<|SPEECH_GENERATION_END|>"
    stopped_generated_text = generated_text.split(stop_token)[0]
    pattern = r"<\|speech_(\d+)\|>"
    matches = findall(pattern, stopped_generated_text)
    speech_tokens = [int(match) for match in matches]
    if len(speech_tokens) < 1:
        raise ValueError(
             "No valid speech tokens found in the output. Generated text:"
             + generated_text[:200]
        )
    audio_array = _decode_speech_tokens(speech_tokens)
    return audio_array

def _decode_speech_tokens(speech_tokens: list[int]) -> ndarray:
    """
    Decode speech tokens using the NeuCodec ONNX decoder.
    """
    if len(speech_tokens) < 1:
        return array([])
    codes = array(speech_tokens, dtype=int32)[None, None, :]
    outputs = decoder.run(None, { "codes": codes })
    return outputs[0][0, 0, :]

def _create_prompt(
    input_text: str,
    reference_codes: ndarray,
    reference_text: str,
    language: str
) -> str:
    """
    Create the speech generation prompt with IPA phonemes.
    """
    ref_phonemes = _convert_to_ipa(reference_text, language=language)
    input_phonemes = _convert_to_ipa(input_text, language=language)
    codes_str = "".join([f"<|speech_{idx.item()}|>" for idx in reference_codes])
    prompt_part1 = "user: Convert the text to speech:<|TEXT_PROMPT_START|>"
    prompt_part2 = f"{ref_phonemes} {input_phonemes}<|TEXT_PROMPT_END|>"
    prompt_part3 = f"\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
    prompt = prompt_part1 + prompt_part2 + prompt_part3
    return prompt

def _convert_to_ipa(
    text: str,
    *,
    language: str
) -> str:
    """
    Convert text to IPA phonemes using the ByT5 ONNX phonemizer model.
    """
    words = text.split()
    word_phonemes = [""]
    for word in words:
        current_word = word
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
        clean_word = ""
        for each_char in current_word:
            if each_char.isalnum():
                clean_word += each_char
        if len(clean_word) == 0:
            if len(sentence_end) > 0:
                word_phonemes.append(sentence_end)
            continue
        prefix = "<" + language + ">: "
        prompt = prefix + clean_word
        encoded_prompt = prompt.encode()
        input_ids = array([list(encoded_prompt)]) + 3
        attention_mask = ones_like(input_ids)
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
        output_ids = [token_id - 3 for token_id in decoder_input_ids]
        phoneme_data = bytes([token_id for token_id in output_ids if 0 <= token_id < 256])
        phonemes = phoneme_data.decode()
        if len(sentence_end) > 0:
            phonemes = phonemes + sentence_end
        word_phonemes.append(phonemes)
    trimmed_word_phonemes = word_phonemes[1:]
    return " ".join(trimmed_word_phonemes)

if __name__ == "__main__":
    import sounddevice as sd
    audio = neutts_nano(
        "It was the best of times. It was the worst of times.",
        voice="dave"
    )
    sd.play(audio, samplerate=24_000)
    sd.wait()
