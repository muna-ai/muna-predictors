#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "muna", "onnx", "onnxruntime", "transformers"]
# ///

from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from numpy import ndarray
from onnx import load, save
from onnx.external_data_helper import convert_model_from_external_data
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
from typing import Annotated, Literal

TaskType = Literal[
    "classification", "clustering", "code retrieval", "document", "fact checking",
    "question answering", "search result", "sentence similarity"
]
TASK_PREFIX_MAP = {
    "classification": f"task: classification | query: ",
    "clustering": f"task: clustering | query: ",
    "code retrieval":  f"task: code retrieval | query: ",
    "document": "title: none | text: ",
    "fact checking": f"task: fact checking | query: ",
    "question answering": f"task: question answering | query: ",
    "search result": f"task: search result | query: ",
    "sentence similarity": f"task: sentence similarity | query: "
}

# Download model graph and weights from HuggingFace
model_id = "onnx-community/embeddinggemma-300m-ONNX"
graph_path = hf_hub_download(
    repo_id=model_id,
    subfolder="onnx",
    filename="model_q4.onnx"
)
weights_path = hf_hub_download(
    repo_id=model_id,
    subfolder="onnx",
    filename="model_q4.onnx_data"
)

# Create merged onnx file from graph and weights files
model_path = "model.onnx"
merged_model = load(graph_path)
convert_model_from_external_data(merged_model)
save(merged_model, model_path)

# Load the model and tokenizer
model = InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "onnx", "onnxruntime", "transformers"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=model, model_path=model_path),
    ]
)
def embedding_gemma(
    texts: Annotated[list[str], Parameter.Generic(description="Input texts to embed.")],
    task: Annotated[TaskType, Parameter.Generic(description="Embedding task type.")]="document"
) -> Annotated[ndarray, Parameter.Embedding(description="Embedding matrix with shape (N,768).")]:
    """
    Embed text with EmbeddingGemma.
    """
    prompts = [TASK_PREFIX_MAP[task] + item for item in texts]
    inputs = tokenizer(prompts, padding=True, return_tensors="np")
    embeddings = model.run(None, inputs.data)[1]
    return embeddings

if __name__ == "__main__":
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    ]
    query_embedding: ndarray = embedding_gemma([query], task="search result")[0]
    document_embeddings = embedding_gemma(documents, task="document")
    similarities = query_embedding @ document_embeddings.T
    print(documents[similarities.argmax()])