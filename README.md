# BESS for KGE in PopTorch

SRAM-only distributed framework for Knowledge Graph Embedding models on IPUs.

## Usage

Tested on Poplar SDK 3.1.0+1205, Ubuntu 20.04, Python 3.8


1\. Install Poplar SDK following the instructions in the Getting Started guide for your IPU system.

2\. Create a virtualenv and activate/install the required packages:
```
source <path to poplar installation>/enable.sh
source <path to popart installation>/enable.sh
python3.8 -m venv .venv
source .venv/bin/activate
pip install $POPLAR_SDK_DIR/poptorch_*.whl
pip install -r requirements.txt 
```

3\. Build custom_ops.so with provided makefile:
```
make all
```


## References
BESS: Balanced Entity Sampling and Sharing for Large-Scale Knowledge Graph Completion ([paper](https://arxiv.org/abs/2211.12281))

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under an MIT license, (see [LICENSE](LICENSE)).
