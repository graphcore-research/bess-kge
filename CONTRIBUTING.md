# How to contribute to the BESS-KGE project

You can contribute to the development of the BESS-KGE project.

## Setup

```shell
python3.8 -m venv .venv
# Add to .venv/bin/activate
# source /PATH_TO_POPLAR_SDK/enable
source .venv/bin/activate
pip install wheel
pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl
pip install -r requirements-dev.txt
```

## Tips

Run `./dev --help` for a list of dev options. In particular, use `./dev ci` to run all CI checks locally.

Run individual tests with pattern matching filtering `./dev tests -k FILTER`.

Add `.cpp` custom ops to `besskge/custom_ops`. Also, update the [Makefile](Makefile) when adding custom ops.
