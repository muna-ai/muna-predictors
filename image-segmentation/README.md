# Image Segmentation Predictors
This directory contains a few predictors that segment objects within an image using popular image segmentation models.

## Running an Image Segmentation Sample
The first step is to run the prediction function directly. First, we recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) as it simplifies working with Python dependencies. Once `uv` is installed, you can run 
any of the image segmentation predictors by simply executing the script directly:
```bash
# Run this in Terminal
$ uv run image-segmentation/yolo_v8_segment_large.py
```

`uv` will automatically install any required Python packages then run the script.

## Compiling the Predictor
Compile the Python function with the Muna CLI:
```bash
# Run this in Terminal
$ muna compile --overwrite image-segmentation/yolo_v8_segment_large.py
```

Muna will generate and compile self-contained, cross-platform code that runs the image segmentation model.

## Running the Predictor
Once compiled, you can run the predictor on any device using our client libraries. For example, run the predictor in 
the command line:
```bash
# Run this in Terminal
$ muna predict @USERNAME/yolo-v8-segment-large --image @path/to/image.jpg
```

> [!TIP]
> Muna compiles predictors to run on Android, iOS, macOS, Linux, visionOS, WebAssembly, and Windows. We provide
> client libraries to run these predictors for JavaScript, Kotlin, Android, React Native, Unity, and more.
> [Learn more](https://docs.muna.ai/predictions/create).