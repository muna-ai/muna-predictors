#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, TensorRTInferenceMetadata
from numpy import ndarray
from torch import float16, inference_mode, int64, ones, Tensor
from torch.export import Dim
from torch.nn import Module
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from typing import Annotated

class Qwen3EmbeddingWrapper(Module):

    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.model(input_ids, attention_mask, use_cache=False).last_hidden_state.float()

# Load model and tokenizer
REPO_ID = "Qwen/Qwen3-Embedding-0.6B"
model = AutoModel.from_pretrained(REPO_ID, dtype=float16)
model = Qwen3EmbeddingWrapper(model).eval()
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, padding_side="left")
_batch = Dim("batch", min=1, max=16)
_seq_len = Dim("seq_len", min=2, max=1024)

@compile(
    tag="@yusuf/qwen-3-embedding-0.6b-trt",
    access="unlisted",
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("transformers"),
    metadata=[
        TensorRTInferenceMetadata(
            model=model,
            exporter="dynamo",
            model_args=[
                ones((8, 512), dtype=int64),
                ones((8, 512), dtype=int64)
            ],
            input_shapes=[
                (_batch, _seq_len),
                (_batch, _seq_len)
            ],
            cuda_arch="sm_80",
            precision="fp16",
        )
    ]
)
@inference_mode()
def qwen_3_embedding_0_6b(
    texts: Annotated[
        list[str],
        Parameter.Generic(description="Input texts to embed.")
    ],
    *,
    instruct: Annotated[
        str,
        Parameter.Generic(description="Task instruction prepended to each text for query embeddings.")
    ]="",
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
    Embed text using Qwen3 Embedding 0.6B.
    """
    prompts = [
        f"Instruct: {instruct}\nQuery:{text}" if instruct else text
        for text in texts
    ]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    last_hidden_state = model(input_ids, attention_mask)
    embeddings = last_hidden_state[:, -1]
    embeddings = embeddings[:, :dimensions]
    embeddings = normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

if __name__ == "__main__":
    task = "Given a web search query, retrieve relevant passages that answer the query"
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
    # Embed queries (with instruction) and documents (without)
    query_embeddings = qwen_3_embedding_0_6b(queries, instruct=task)
    doc_embeddings = qwen_3_embedding_0_6b(documents)
    scores = query_embeddings @ doc_embeddings.T
    print(f"Similarity scores:\n{scores}")