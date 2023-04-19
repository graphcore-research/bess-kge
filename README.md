# BESS for KGE in PopTorch

A library for Knowledge Graph Embedding models on IPU implementing the distribution framework [BESS](https://arxiv.org/abs/2211.12281), with embedding tables stored in IPU SRAM.

See the [documentation](https://symmetrical-adventure-69267rm.pages.github.io/).

## Usage

Tested on Poplar SDK 3.2.0+1277, Ubuntu 20.04, Python 3.8

1\. Install Poplar SDK following the instructions in the Getting Started guide for your IPU system. More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


2\. Create a virtualenv with PopTorch:
```
source <path to poplar installation>/enable.sh
source <path to popart installation>/enable.sh
python3.8 -m venv .venv
source .venv/bin/activate
pip install wheel
pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl
```

3\. Pip install BESS-KGE:
```
pip install git+ssh://git@github.com/graphcore-research/bess-kge.git
```

4\. Import and use:
```
import besskge
```

## Tutorials and examples

For a walkthrough of the library functionalities, see our jupyter notebooks (better if in sequence): 
1. [KGE training and inference on the OGBL-BioKG dataset](notebooks/1_biokg_training_inference.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/bess-kge?container=graphcore%2Fpytorch-jupyter%3A3.2.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fnotebooks%2F1_biokg_training_inference.ipynb)
2. [Link prediction on the YAGO3-10 dataset](notebooks/2_yago_topk_prediction.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/bess-kge?container=graphcore%2Fpytorch-jupyter%3A3.2.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fnotebooks%2F2_yago_topk_prediction.ipynb)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## References
BESS: Balanced Entity Sampling and Sharing for Large-Scale Knowledge Graph Completion ([arXiv](https://arxiv.org/abs/2211.12281))

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under an MIT license, (see [LICENSE](LICENSE)).

The OGBL-BioKG Dataset is licensed under a CC-0 license.

See [requirements.txt](requirements.txt) and [requirements-dev.txt](requirements-dev.txt) for dependencies.
