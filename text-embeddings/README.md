# Text Embedding Predictors
This directory contains a few predictors that create embedding vectors from text using popular text embedding models.

## Running a Text Embedding Sample
The first step is to run the prediction function directly. First, we recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) as it simplifies working with Python dependencies. Once `uv` is installed, you can run 
any of the text embedding predictors by simply executing the script directly:
```bash
# Run this in Terminal
$ uv run text-embedding/embedding_gemma.py
```

`uv` will automatically install any required Python packages then run the script.

## Compiling the Predictor
Compile the Python function with the Muna CLI:
```bash
# Run this in Terminal
$ muna compile --overwrite text-embedding/embedding_gemma.py
```

Muna will generate and compile a self-contained executable binary that generates embeddings from input text.

> [!TIP]
> You can also use pre-compiled text-to-speech predictors on [Muna](https://muna.ai/explore).

## Running the Predictor
Once compiled, you can run the predictor on any device using our client libraries. For example, run the predictor in 
the command line:
```bash
# Run this in Terminal
$ muna predict @USERNAME/embedding-gemma --texts "What is the capital of France?"
```

> [!TIP]
> Muna compiles predictors to run on Android, iOS, macOS, Linux, visionOS, WebAssembly, and Windows. We provide
> client libraries to run these predictors for JavaScript, Kotlin, Android, React Native, Unity, and more.
> [Learn more](https://docs.muna.ai/predictions/create).

## Using the OpenAI Embedding API
Muna's client libraries for [Python](https://github.com/muna-ai/muna-py) and [JavaScript](https://github.com/muna-ai/muna-js) 
provide a mock OpenAI client that can be used to run compiled predictors, allowing for easy migration to Muna:

```js
// 💥 Create a Muna client
const muna = new Muna();
const openai = muna.beta.openai;

// 🔥 Create embeddings
const embeddings = await openai.embeddings.create({
    model: "@USERNAME/embedding-gemma",
    input: "The quick brown fox jumped over the lazy dog.",
    encoding_format: "float",
});

// 🚀 Use the results
retrievalAugmentedGeneration(embeddings);
```