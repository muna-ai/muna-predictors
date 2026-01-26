# Muna Compiler
![Muna logo](https://raw.githubusercontent.com/muna-ai/.github/main/logo_wide.png)

Muna compiles stateless Python functions to run anywhere.
This project is a playground for testing the Muna compiler. Over time, we expect to open-source more 
and more of the compiler infrastructure in this project.

## Setup Instructions
First, install Muna for Python:
```sh
# Run this in Terminal
$ pip install --upgrade muna
```

Next, head over to the [Muna dashboard](https://muna.ai/settings/developer) to generate an access key. 
Once generated, sign in to the CLI:
```sh
# Login to the Muna CLI
$ muna auth login <ACCESS KEY>
```

## Compiling a Function
The [`predictors`](/predictors) directory contains several prediction functions, ranging from very simple functions to 
AI inference with PyTorch. Internally, we use these functions to test language and library coverage in the compiler.
Use the Muna CLI to compile the function, providing the path to the module where the function is defined:
```sh
# Compile the decorated function at the module path
$ muna compile --overwrite path/to/module.py
```

The compiler will load the entrypoint function, create a remote sandbox, and compile the function:

![compiling a function](media/fma.gif)

## Useful Links
- [Discover predictors to use in your apps](https://muna.ai/explore).
- [Join our Slack community](https://muna.ai/slack).
- [Check out our docs](https://docs.muna.ai).
- Learn more about us [on our blog](https://blog.muna.ai).
- Reach out to us at [hi@muna.ai](mailto:hi@muna.ai).

Muna is a product of [NatML Inc](https://github.com/natmlx).
