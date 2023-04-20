# For developers

Dev setup:

```shell
python3.8 -m venv .venv
# Add to .venv/bin/activate
# source /PATH_TO_POPLAR_SDK/enable
source .venv/bin/activate
pip install wheel
pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl
pip install -r requirements-dev.txt
```

Run `./dev --help` for a list of dev options. In particular, use `./dev ci` to run all CI checks locally. 

Individual tests can be run with pattern matching filtering `./dev tests -k FILTER`.

`.cpp` custom ops are to be added in `besskge/custom_ops`; when doing so, also update the [Makefile](Makefile).
