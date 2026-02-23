#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "muna", "onnxruntime", "sounddevice==0.5.2"]
# ///

from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, OnnxRuntimeInferenceSessionMetadata
from numpy import array, float32, int64, load, ndarray, ones_like
from onnxruntime import InferenceSession
from typing import Annotated, Literal

GenerationVoice = Literal[
    "Bella", "Jasper", "Luna", "Bruno",
    "Rosie", "Hugo", "Kiki", "Leo"
]

VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}

# Download KittenTTS Mini v0.8
kitten_model_path = hf_hub_download(
    repo_id="KittenML/kitten-tts-mini-0.8",
    filename="kitten_tts_mini_v0_8.onnx"
)
kitten_voices_path = hf_hub_download(
    repo_id="KittenML/kitten-tts-mini-0.8",
    filename="voices.npz"
)

# Download ByT5-based grapheme to phoneme converter
byt5_model_path = hf_hub_download(
    repo_id="OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx",
    filename="fdemelo_g2p-mbyt5-12l-ipa-childes-espeak.onnx"
)

# Load models and voices
kitten_model = InferenceSession(kitten_model_path)
byt5_model = InferenceSession(byt5_model_path)
voices = load(kitten_voices_path)

def _create_word_index_dictionary():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    return { ord(c): i for i, c in enumerate(symbols) }

IPA_CODE_MAP = _create_word_index_dictionary()

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "onnxruntime"),
    targets=["ios", "macos"],
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(
            session=kitten_model,
            model_path=kitten_model_path,
            providers=["cpu"]
        ),
        OnnxRuntimeInferenceSessionMetadata(
            session=byt5_model,
            model_path=byt5_model_path,
            providers=["cpu"]
        ),
    ]
)
def kitten_tts_mini_0_8(
    text: Annotated[
        str,
        Parameter.Generic(description="Text to generate speech from.")
    ],
    *,
    voice: Annotated[
        GenerationVoice,
        Annotations.AudioVoice(description="Generation voice.")
    ],
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
    Perform text-to-speech with Kitten TTS Mini v0.8.
    """
    voice_id = VOICE_ALIASES[voice]
    ipa = _convert_to_ipa(text)
    tokens = [IPA_CODE_MAP[ord(c)] for c in ipa]
    tokens.insert(0, 0)
    tokens.append(0)
    input_ids = array(tokens, dtype=int64)[None]
    ref_id = min(len(text), voices[voice_id].shape[0] - 1)
    ref_s = voices[voice_id][ref_id:ref_id + 1]
    speed_spec = array([speed], dtype=float32)
    outputs = kitten_model.run(None, {
        "input_ids": input_ids,
        "style": ref_s,
        "speed": speed_spec
    })
    audio = outputs[0].squeeze()[:-5000]
    return audio

def _convert_to_ipa(text: str) -> str:
    """
    Convert text to IPA phonemes using ByT5-G2P.
    """
    prompt = f"<en-US>: {text}"
    num_special_tokens = 3
    input_ids = array([list(prompt.encode())]) + num_special_tokens
    attention_mask = ones_like(input_ids)
    eos_token_id = 1
    max_length = 510
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
    output_ids = [token_id - num_special_tokens for token_id in decoder_input_ids]
    phoneme_data = bytes([token_id for token_id in output_ids if 0 <= token_id < 256])
    phonemes = phoneme_data.decode()
    return phonemes

if __name__ == "__main__":
    import sounddevice as sd
    audio = kitten_tts_mini_0_8(
        text="Hello! This is a test of the Kitten TTS mini model.",
        voice="Jasper"
    )
    sd.play(audio, samplerate=24_000)
    sd.wait()