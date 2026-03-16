#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceMetadata
from numpy import ndarray
from torch import clamp, inference_mode, int64, ones, sum, Tensor
from torch.export import Dim
from torch.nn import Module
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from typing import Annotated

class MiniLMWrapper(Module):

    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.model(input_ids, attention_mask).last_hidden_state

# Create model and tokenizer
REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
model = MiniLMWrapper(AutoModel.from_pretrained(REPO_ID)).eval()
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
_batch = Dim("batch_size", min=1, max=64)
_seq_len = Dim("seq_len", min=2, max=512)

@compile(
    tag="@sentence-transformers/all-minilm-l6-v2",
    access="public",
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("transformers"),
    metadata=[
        OnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[
                ones((2, 128), dtype=int64),
                ones((2, 128), dtype=int64)
            ],
            input_shapes=[
                (_batch, _seq_len),
                (_batch, _seq_len),
            ],
        )
    ]
)
@inference_mode()
def all_minilm_l6_v2(
    texts: Annotated[
        list[str],
        Parameter.Generic(description="Input texts to embed.")
    ]
) -> Annotated[
    ndarray,
    Parameter.Embedding(description="Embedding matrix.")
]:
    """
    Embed text using all-MiniLM-L6-v2.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_embeddings = model(input_ids, attention_mask)
    embeddings = _mean_pooling(token_embeddings, attention_mask)
    final_embeddings = normalize(embeddings, p=2, dim=1)
    return final_embeddings.numpy()

def _mean_pooling(token_embeddings: Tensor, attention_mask: Tensor):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    return (
        sum(token_embeddings * input_mask_expanded, dim=1) /
        clamp(input_mask_expanded.sum(1), min=1e-9)
    )

if __name__ == "__main__":
    embeddings = all_minilm_l6_v2([
        "This is an example sentence.",
        "Each sentence is converted to a vector."
    ])
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First 20 values: {embeddings[0][:20]}")
