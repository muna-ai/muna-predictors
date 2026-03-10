#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub", "muna", "onnx", "onnxruntime",
#   "torch", "transformers"
# ]
# ///

from huggingface_hub import snapshot_download
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, OnnxRuntimeInferenceSessionMetadata
from numpy import ndarray
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from onnxruntime import InferenceSession
from pathlib import Path
from torch import from_numpy
from torch.nn.functional import normalize
from transformers import AutoTokenizer
from typing import Annotated

# Download model and model data
REPO_ID = "perplexity-ai/pplx-embed-v1-0.6b"
model_dir = snapshot_download(REPO_ID, allow_patterns=[
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "onnx/model.onnx_data_1"
])
model_path = Path(model_dir) / "onnx/model_consolidated.onnx"
data_path = Path(model_dir) / "onnx/model_consolidated.onnx.data"

# Consolidate data files into one
if not data_path.exists():
    model = onnx.load(
        Path(model_dir) / "onnx/model.onnx",
        load_external_data=True
    )
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_path.name
    )
    onnx.save(model, model_path)
    del model

# Load model session and tokenizer
session = InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

@compile(
    tag="@perplexity/pplx-embed-v1-0.6b",
    access="public",
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("onnx", "onnxruntime", "transformers"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(
            session=session,
            model_path=model_path,
            external_data_path=data_path
        )
    ]
)
def pplx_embed_v1_0_6b(
    texts: Annotated[
        list[str],
        Parameter.Generic(description="Input texts to embed.")
    ],
    *,
    dimensions: Annotated[int, Annotations.EmbeddingDims(
        description="Embedding dimensions.",
        min=32,
        max=1024
    )]=1024
) -> Annotated[
    ndarray,
    Parameter.Embedding(description="Embedding matrix.")
]:
    """
    Embed text using Perplexity pplx-embed-v1-0.6B.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32768,
        return_tensors="np"
    )
    embeddings = session.run(None, {
        "input_ids": encoded.data["input_ids"],
        "attention_mask": encoded.data["attention_mask"],
    })[1]
    embeddings = embeddings[:,:dimensions]
    embeddings = normalize(from_numpy(embeddings), p=2, dim=1)
    return embeddings.numpy()

if __name__ == "__main__":
    queries = [
        "What is the capital of China?",
        "Explain gravity"
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the "
        "movement of planets around the sun.",
    ]
    query_embeddings = pplx_embed_v1_0_6b(queries)
    doc_embeddings = pplx_embed_v1_0_6b(documents)
    scores = query_embeddings @ doc_embeddings.T
    print(f"Shape: {query_embeddings.shape}")
    print(f"Similarity scores:\n{scores}")