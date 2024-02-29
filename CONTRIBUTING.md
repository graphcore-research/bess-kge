# How to contribute to the BESS-KGE project

![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)

You can contribute to the development of the BESS-KGE project, even if you don't have access to IPUs (you can use the [IPUModel](https://docs.graphcore.ai/projects/poptorch-user-guide/en/3.2.0/reference.html#poptorch.Options.useIpuModel) to emulate most functionalities of the physical hardware).

## Setup on local machine

To develop on a local machine, first install the Poplar SDK following the instructions in the [Getting Started guide for your IPU system](https://docs.graphcore.ai/en/latest/getting-started.html#getting-started).

Then, enable the Poplar SDK, create and activate a Python `virtualenv` and install the PopTorch wheel and all the necessary dependencies: 

```shell
python3.8 -m venv .venv
# Add to .venv/bin/activate
# source /PATH_TO_POPLAR_SDK/enable
source .venv/bin/activate
```
```shell
pip install wheel
pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl
pip install -r requirements-dev.txt
```

Finally, clone your fork of the BESS-KGE repository and build all custom ops by running `./dev build`

## Development tips

The `./dev` command can be used to run several utility scripts during development. Check `./dev --help` for a list of dev options.

Before submitting a PR to the upstream repo, use `./dev ci` to run all CI checks locally. In particular, be mindful of our formatting requirements: you can check for formatting errors by running `./dev format` and `./dev lint` (both commands are automatically run inside `./dev ci`).

Add unit tests to the `tests` folder. You can run individual unit tests with pattern matching filtering `./dev tests -k FILTER`.

Add `.cpp` custom ops to `besskge/custom_ops`. Also, update the [Makefile](Makefile) when adding custom ops.
