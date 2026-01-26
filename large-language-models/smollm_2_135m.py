#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "llama-cpp-python", "muna"]
# ///

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from muna import compile, Parameter, Sandbox
from muna.beta import LlamaCppInferenceMetadata
from muna.beta.openai import Annotations, ChatCompletionChunk, Message
from typing import Annotated, Iterator

model_path = hf_hub_download(
    "unsloth/SmolLM2-135M-Instruct-GGUF",
    "SmolLM2-135M-Instruct-Q8_0.gguf"
)
model = Llama(model_path=model_path, verbose=False)

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "llama-cpp-python"),
    metadata=[
        LlamaCppInferenceMetadata(model=model)
    ]
)
def smollm_2_135m(
    messages: Annotated[
        list[Message],
        Parameter.Generic(description="Messages comprising the chat conversation so far.")
    ],
    *,
    max_output_tokens: Annotated[
        int,
        Annotations.MaxOutputTokens(description="Maximum number of tokens in the response.")
    ]=5_000,
) -> Iterator[ChatCompletionChunk]:
    """
    Generate text with HuggingFace SmolLM2 135M.
    """
    for chunk in model.create_chat_completion(
        messages=messages,
        max_tokens=max_output_tokens,
        stream=True
    ):
       yield chunk

if __name__ == "__main__":
    stream = smollm_2_135m([
        Message(role="system", content="You are an AI assistant that provides very brief answers."),
        Message(role="user", content="What is the capital of France.")
    ])
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta.get("content"), end="")
    print()