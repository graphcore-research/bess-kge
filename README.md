# BESS-KGE
![Continuous integration](https://github.com/graphcore-research/bess-kge/actions/workflows/ci.yaml/badge.svg)

[**Install guide**](#usage)
| [**Tutorials**](#paperspace-notebook-tutorials)
| [**Documentation**](https://symmetrical-adventure-69267rm.pages.github.io/)


BESS-KGE is a PyTorch library for Knowledge Graph Embedding models on IPU implementing the distribution framework [BESS](https://arxiv.org/abs/2211.12281), with embedding tables stored in IPU SRAM.

## Features and limitations



### BESS distribution framework
### Modules

| API | Functions 
| --- | --- |
| [`besskge.dataset`](besskge/dataset.py) | Build, save and load KG datasets as collections of (h,r,t) triples.|
| [`besskge.sharding`](besskge/sharding.py) | Shard embedding tables and triple sets for distributed execution.|
| [`besskge.embedding`](besskge/embedding.py) | Utilities to initialize entity and relation embedding tables.|
| [`besskge.negative_sampler`](besskge/negative_sampler.py) | Sample entities to use as corrupted heads/tails when constructing negative samples. Negative entities can be sampled randomly, based on entity type or based on the triple to corrupt.|
| [`besskge.batch_sampler`](besskge/batch_sampler.py) | Sample batches of positive and negative triples for each processing device, according to the BESS distribution scheme.|
| [`besskge.scoring`](besskge/scoring.py) | Functions used to score positive and negative triples for different KGE models, e.g. TransE, ComplEx, RotatE, DistMult.|
| [`besskge.loss`](besskge/loss.py) | Functions used to compute the batch loss based on positive and negative scores, e.g. log-sigmoid loss, margin ranking loss.|
| [`besskge.metric`](besskge/metric.py) | Functions used to compute metrics for the predictions of KGE models, e.g. MRR, Hits@K.|
| [`besskge.bess`](besskge/bess.py) | PyTorch modules implementing the BESS distribution scheme for KGE training and inference on multiple IPUs. |
| [`besskge.utils`](besskge/utils.py) | General puropose utilities.|

### Known limitations

* BESS-KGE supports distribution up to 16 IPUs.
* Storing embeddings in SRAM introduces limitations on the size of the embedding tables, and therefore on the entity count in the KG. Using the optimizer Adam, float32 weights and an embedding size of 128, this limit can be quantified in ~4M entities when sharding tables across 16 IPUs.
* `besskge.bess.TopKQueryBessKGE` currently cannot be used with distance-based scoring functions (e.g. TransE, RotatE).

## Usage

Tested on Poplar SDK 3.2.0+1277, Ubuntu 20.04, Python 3.8

1\. Install Poplar SDK following the instructions in the Getting Started guide for your IPU system. More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).

2\. Create a virtualenv with PopTorch:
```shell
source <path to Poplar installation>/enable.sh
source <path to PopART installation>/enable.sh
python3.8 -m venv .venv
source .venv/bin/activate
pip install wheel
pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl
```

3\. Pip install BESS-KGE:
```shell
pip install git+ssh://git@github.com/graphcore-research/bess-kge.git
```

4\. Import and use:
```python
import besskge
```

## Paperspace notebook tutorials

For a walkthrough of the library functionalities, see our jupyter notebooks (better if in the suggested sequence): 
1. [KGE training and inference on the OGBL-BioKG dataset](notebooks/1_biokg_training_inference.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/bess-kge?container=graphcore%2Fpytorch-jupyter%3A3.2.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fnotebooks%2F1_biokg_training_inference.ipynb)
2. [Link prediction on the YAGO3-10 dataset](notebooks/2_yago_topk_prediction.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/bess-kge?container=graphcore%2Fpytorch-jupyter%3A3.2.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fnotebooks%2F2_yago_topk_prediction.ipynb)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## References
BESS: Balanced Entity Sampling and Sharing for Large-Scale Knowledge Graph Completion ([arXiv](https://arxiv.org/abs/2211.12281))

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under an MIT license, (see [LICENSE](LICENSE)).

See [NOTICE.md](NOTICE.md) for dependencies and further details.