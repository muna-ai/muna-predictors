#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "muna", "onnxruntime", "sounddevice"]
# ///

from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from muna.beta.openai import Annotations
from numpy import array, float32, int64, load, ndarray, ones_like
from onnxruntime import InferenceSession
from typing import Annotated, Literal

GenerationLanguage = Literal[
    "ca-ES", "cy-GB", "da-DK", "de-DE", "en-US", "en-GB", "es-ES",
    "et-EE", "eu-ES", "fa-IR", "fr-FR", "ga-IE", "hr-HR", "hu-HU",
    "id-ID", "is-IS", "it-IT", "ja-JP", "ko-KR", "nb-NO", "nl-NL",
    "pl-PL", "pt-BR", "pt-PT", "qu-PE", "ro-RO", "sr-RS", "sv-SE",
    "tr-TR", "yue-CN", "zh-CN"
]
GenerationVoice = Literal[
    "expr-voice-2-m", "expr-voice-2-f",
    "expr-voice-3-m", "expr-voice-3-f",
    "expr-voice-4-m", "expr-voice-4-f",
    "expr-voice-5-m", "expr-voice-5-f"
]

# Download models
kitten_model_path = hf_hub_download(
    repo_id="KittenML/kitten-tts-nano-0.2",
    filename="kitten_tts_nano_v0_2.onnx"
)
kitten_voices_path = hf_hub_download(
    repo_id="KittenML/kitten-tts-nano-0.2",
    filename="voices.npz"
)
byt5_model_path = hf_hub_download(
    "OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx",
    "fdemelo_g2p-mbyt5-12l-ipa-childes-espeak.onnx"
)

# Load model and voices
kitten_model = InferenceSession(kitten_model_path)
byt5_model = InferenceSession(byt5_model_path)
voices = load(kitten_voices_path)

def _create_word_index_dictionary():
    """
    Create word to index mapping for text cleaning.
    """
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len(symbols)):
        dicts[ord(symbols[i])] = i
    return dicts

IPA_CODE_MAP = _create_word_index_dictionary()

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "onnxruntime"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=kitten_model, model_path=kitten_model_path),
        OnnxRuntimeInferenceSessionMetadata(session=byt5_model, model_path=byt5_model_path),
    ]
)
def kitten_tts(
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
        GenerationLanguage,
        Parameter.Generic(description="Generation language.")
    ]="en-US",
    speed: Annotated[float, Annotations.AudioSpeed(
        description="Voice speed multiplier.",
        min=0.25,
        max=2.
    )]=1.0
) -> Annotated[ndarray, Parameter.Audio(
    description="Linear PCM audio samples with shape (F,) and sample rate 24KHz.",
    sample_rate=24_000
)]:
    """
    Perform text-to-speech with Kitten TTS.
    """
    # Convert to IPA
    prompt_ipa = _convert_to_ipa(text, language=language)
    # Build input token list kitten expects
    tokens = [IPA_CODE_MAP[ord(item)] for item in prompt_ipa]
    tokens.insert(0, 0)
    tokens.append(0)
    input_ids = array(tokens, dtype=int64)[None]
    # Specify speed
    speed_spec = array([speed], dtype=float32)
    # Run the model
    outputs = kitten_model.run(None, {
        "input_ids": input_ids,
        "style": voices[voice],
        "speed": speed_spec
    })
    # Trim audio
    audio = outputs[0][100:-100]
    # Return speech
    return audio

def _convert_to_ipa(
    text: str,
    *,
    language: GenerationLanguage
) -> str:
    """
    Convert a string to an IPA string with ByT5-G2P.
    """
    # Tokenize input
    prompt = f"<{language}>: {text}"
    num_special_tokens = 3
    input_ids = array([list(prompt.encode())]) + num_special_tokens
    attention_mask = ones_like(input_ids)
    # Generate autoregressively
    eos_token_id = 1
    max_length = 100 # make this a default argument once we get to #89
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
    output_ids = [token_id - num_special_tokens for token_id in decoder_input_ids]
    phoneme_data = bytes([token_id for token_id in output_ids if 0 <= token_id < 256])
    phonemes = phoneme_data.decode()
    # Return
    return phonemes

if __name__ == "__main__":
    import sounddevice as sd
    # Generate audio
    audio = kitten_tts(
        text="Kitten is such an odd model.",
        voice="expr-voice-3-m"
    )
    # Playback
    sd.play(audio, samplerate=24_000)
    sd.wait()