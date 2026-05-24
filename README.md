# hymenoptera_dino_cls project
<hr>

## Installation
`git clone ...` - clone the package to your host.
`uv sync` - install all the dependencies.

## Train
`uv run src/train.py` - strat training process with the parameters given in the hydra config sections *dataset* and *model*.

## MLFlow serving
`uv run src/scripts/run_server` - run MLFlow Serving endpoints at the host and port definded in the hydra config section *tracking*. By default the host is **127.0.0.1** and the port is **8080**.

## GUI inference
`uv run streamlit run src/scripts/run_gui.py` - run streamlit GUI for manual access to the inference endpoint. By default GUI runs on host **127.0.0.1** and port **8501** - check your console output for the actual URL.