# BESS-KGE
![Continuous integration](https://github.com/graphcore-research/bess-kge/actions/workflows/ci.yaml/badge.svg)

[**Install guide**](#usage)
| [**Tutorials**](#paperspace-notebook-tutorials)
| [**Documentation**](https://symmetrical-adventure-69267rm.pages.github.io/)


BESS-KGE is a PyTorch library for Knowledge Graph Embedding models on IPU implementing the distribution framework [BESS](https://arxiv.org/abs/2211.12281), with embedding tables stored in IPU SRAM.

## Features and limitations

Shallow KGE models are typically memory-bound, as little compute needs to be performed to score (h,r,t) triples once the embeddings of entities and relation types used in the batch have been retrieved. 
BESS (Balanced Entity Sampling and Sharing) is a KGE distribution framework designed to maximize bandwith for gathering embeddings, by 
* storing them in fast-access IPU on-chip memory;
* minimizing communication time for sharing embeddings between workers, leveraging balanced collective operators over high-bandwidth IPU-links.

This allows BESS-KGE to achieve high throughput for both training and inference.

### BESS overview

When distributing the workload over $n$ workers (=IPUs), BESS randomly splits the entity embedding table in $n$ shards of equal size, each of which is stored in one of the workers' memory. The embedding table for relation types, on the other hand, is replicated across workers, as it is usually much smaller.

<div align="center">
<figure>
  <img src="docs/source/images/embedding_sharding.jpg" height=250>
  <figcaption>
  
  **Figure 1**. Entity table sharding across $n=3$ workers.
  
  </figcaption>
</figure>
</div>

The entity sharding induces a partitioning of the triples in the dataset, according to the shardpair of head entity and tail entity. At execution time (for both training and inference) batches are constructed by sampling triples uniformly from each of the $n^2$ shardpairs. Negative entities, used to corrupt the head or tail of a triple to construct negative samples, are also sampled in a balanced way to ensure a variety that is beneficial to the final embedding quality.

<div id="figure2" align="center">
<figure>
  <img src="docs/source/images/batch_together.jpg" width=700>
  <figcaption>
  
  **Figure 2**. *Left*: a batch is made of $n^2=9$ blocks, each containing the same number of triples. The head embeddings of triples in block $(i,j)$ are stored on worker $i$, the tail embeddings on worker $j$, for $i,j = 0,1,2$. *Right*: the negative entities used to corrupt triples in block $(i,j)$ are sampled in equal number from all of the $n$ shards. In this example, negative samples are constructed by corrupting tails.
  
  </figcaption>
</figure>
</div>

This batch cook-up scheme allows us to balance workload and communication across workers. First, each worker needs to gather the same number of embeddings from its on-chip memory, both for positive and negative samples. These include the embeddings neeeded by the worker itself, and the embeddings needed by its peers.

<div align="center">
<figure>
  <img src="docs/source/images/gather.jpg" width=650>
  <figcaption>
  
  **Figure 3**. The required embeddings are gathered from the IPUs' SRAM. Each worker needs to retrieve the head embeddings for $n$ positive triple blocks, and the same for tail embeddings (the $3 + 3$ triangles of same colour in [Figure 2 (left)](#figure2)). In addition to that, the worker gathers the portion (=$1/3$) stored in its memory of the negative tails needed by all of the $n^2$ blocks.
  
  </figcaption>
</figure>
</div>

The batch in [Figure 2](#figure2) can then be reconstrcuted by sharing the embeddings of positive **tails** and negative entities between workers through a balanced AllToAll collective operator. Head embeddings remain inplace, as each triple block is then scored on the worker where the head embedding is stored.

<div align="center">
<figure>
  <img src="docs/source/images/alltoall.jpg" width=650>
  <figcaption>
  
  **Figure 4**. Embeddings of positive and negative tails are exchanged between workers with an AllToAll collective (red arrows), which effectively transposes rows and columns of the $n^2$ blocks in the picture. After this exchange, each worker (vertical column) has the correct $n$ blocks of positive triples and $n$ blocks of negative tails to compute positive and negative scores.
  
  </figcaption>
</figure>
</div>

Additional variations of the BESS distribution scheme are detailed in the [documentation](https://symmetrical-adventure-69267rm.pages.github.io/API_ref/bess.html).

### Modules

All APIs are documented [here](https://symmetrical-adventure-69267rm.pages.github.io/).

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
* Storing embeddings in SRAM introduces limitations on the size of the embedding tables, and therefore on the entity count in the knowledge graph. Some (approximate) estimates for these limitations are given in the table below (assuming FP16 for weights and FP32 for gradient accumulation and second order momentum). Notice that the cap will also depend on the batch size and number of negative samples used.

<table>
<thead>
  <tr>
    <th colspan="2" style="text-align: center">Embeddings</th>
    <th rowspan="2">Optimizer</th>
    <th rowspan="2">Gradient<br>accumulation</th>
    <th colspan="2" style="text-align: center">Max number of entities <br>(# embedding parameters) on</th>
  </tr>
  <tr>
    <th>size</th>
    <th>dtype</th>
    <th>IPU-POD4</th>
    <th>IPU-POD16</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>100</td>
    <td>float16</td>
    <td>SGDM</td>
    <td>No</td>
    <td>3.2M (3.2e8)</td>
    <td>13M (1.3e9)</td>
  </tr>
  <tr>
    <td>128</td>
    <td>float16</td>
    <td>Adam</td>
    <td>No</td>
    <td>2.4M (3.0e8)</td>
    <td>9.9M (1.3e9)</td>
  </tr>
  <tr>
    <td>256</td>
    <td>float16</td>
    <td>SGDM</td>
    <td>Yes</td>
    <td>900K (2.3e8)</td>
    <td>3.5M (9.0e8)</td>
  </tr>
  <tr>
    <td>256</td>
    <td>float16</td>
    <td>Adam</td>
    <td>No</td>
    <td>1.2M (3.0e8)</td>
    <td>4.8M (1.2e9)</td>
  </tr>
  <tr>
    <td>512</td>
    <td>float16</td>
    <td>Adam</td>
    <td>Yes</td>
    <td>375K (1.9e8)</td>
    <td>1.5M (7.7e8)</td>
  </tr>
</tbody>
</table>

If getting an error message during compilation about the onnx protobuffer exceeding maximum size, we recommend saving weights to a file using the `poptorch.Options` API `options._Popart.set("saveInitializersToFile", "my_file.onnx")`.

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
3. [FP16 weights and compute on the OGBL-WikiKG2 dataset](notebooks/3_wikikg2_fp16.ipynb) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/bess-kge?container=graphcore%2Fpytorch-jupyter%3A3.2.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fnotebooks%2F3_wikikg2_fp16.ipynb)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## References
BESS: Balanced Entity Sampling and Sharing for Large-Scale Knowledge Graph Completion ([arXiv](https://arxiv.org/abs/2211.12281))

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under MIT license, (see [LICENSE](LICENSE)).

See [NOTICE.md](NOTICE.md) for dependencies, credits, derived work and further details.