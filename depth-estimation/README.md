# Depth Estimation Predictors
This folder contains a few predictors that estimate metric depth using popular monocular depth estimation models.

## Running a Depth Estimation Sample
The first step is to run the prediction function directly. First, we recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) as it simplifies working with Python dependencies. Once `uv` is installed, you can run 
any of the depth estimation predictors by simply executing the script directly:
```bash
# Run this in Terminal
$ uv run depth-estimation/apple_depth_pro.py
```

`uv` will automatically install any required Python packages then run the script.

> [!WARNING]
> Some depth estimation predictors require that the model's original repository is cloned. Refer to the 
> Python scripts for more information.

## Compiling the Predictor
Cmpile the Python function with the Muna CLI:
```bash
# Run this in Terminal
$ muna compile --overwrite depth-estimation/apple_depth_pro.py
```

Muna will generate and compile self-contained native code (C++, Rust, etc) that runs the depth estimation model.

## Running the Predictor
Once compiled, you can run the predictor on any device using our client libraries. For example, run the predictor in 
the command line:
```bash
# Run this in Terminal
$ muna predict @USERNAME/depth-pro --image @path/to/image.jpg
```

> [!TIP]
> Muna compiles predictors to run on Android, iOS, macOS, Linux, visionOS, WebAssembly, and Windows. We provide
> client libraries to run these predictors for JavaScript, Kotlin, Android, React Native, Unity, and more.
> [Learn more](https://docs.muna.ai/predictions/create).