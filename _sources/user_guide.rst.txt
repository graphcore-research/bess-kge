User guide
================

Installation and usage
------------------------

1. Install the Poplar SDK following the instructions in the
`Getting Started guide for your IPU system <https://docs.graphcore.ai/en/latest/getting-started.html#getting-started>`_.

2. Enable the Poplar SDK, create and activate a Python :code:`virtualenv` and install the PopTorch wheel:

.. code-block::

    source <path to Poplar installation>/enable.sh
    source <path to PopART installation>/enable.sh
    python3.8 -m venv .venv
    source .venv/bin/activate
    pip install wheel
    pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl


More details are given in the
`PyTorch quick start guide <https://docs.graphcore.ai/projects/pytorch-quick-start>`_.

3. Pip install BESS-KGE:

.. code-block::

    pip install git+https://github.com/graphcore-research/bess-kge.git

4. Import and use:

.. code-block::

    import besskge

.. Note:: The library has been tested on Poplar SDK 3.3.0+1403, Ubuntu 20.04, Python 3.8.


Getting started
------------------------

For a walkthrough of the main :code:`besskge` library functionalities, see our Jupyter notebooks.
We recommend the following sequence:

1. `KGE training and inference on the OGBL-BioKG dataset <https://github.com/graphcore-research/bess-kge/tree/main/notebooks/1_biokg_training_inference.ipynb>`_.
2. `Link prediction on the YAGO3-10 dataset <https://github.com/graphcore-research/bess-kge/tree/main/notebooks/2_yago_topk_prediction.ipynb>`_.
3. `FP16 weights and compute on the OGBL-WikiKG2 dataset <https://github.com/graphcore-research/bess-kge/tree/main/notebooks/3_wikikg2_fp16.ipynb>`_.

.. |run_on_gradient| image:: ../gradient-badge.svg 

Click on the |run_on_gradient| button inside the notebooks to run them **for free** on physical IPUs
available on `Paperspace <https://www.paperspace.com/graphcore>`_.

Limitations
------------------------

* :code:`besskge` supports distribution for up to 16 IPUs.
* Storing embeddings in SRAM introduces limitations on the size of the embedding tables,
  and therefore on the entity count in the knowledge graph. Some (approximate) estimates for these limitations
  are given in the table below (assuming FP16 for weights and FP32 for gradient accumulation and second order momentum).
  Notice that the cap will also depend on the batch size and the number of negative samples used.

+----------------+-----------+--------------+-----------------------------+
|   Embeddings   |           |              |   Max number of entities    |
|                | Optimizer | Gradient     | (# embedding parameters) on |
+------+---------+           | accumulation +--------------+--------------+
| size |  dtype  |           |              |   IPU-POD4   |   IPU-POD16  |
+------+---------+-----------+--------------+--------------+--------------+
| 100  | float16 | SGDM      | No           | 3.2M (3.2e8) | 13M (1.3e9)  |
+------+---------+-----------+--------------+--------------+--------------+
| 128  | float16 | Adam      | No           | 2.4M (3.0e8) | 9.9M (1.3e9) |
+------+---------+-----------+--------------+--------------+--------------+
| 256  | float16 | SGDM      | Yes          | 900K (2.3e8) | 3.5M (9.0e8) |
+------+---------+-----------+--------------+--------------+--------------+
| 256  | float16 | Adam      | No           | 1.2M (3.0e8) | 4.8M (1.2e9) |
+------+---------+-----------+--------------+--------------+--------------+
| 512  | float16 | Adam      | Yes          | 375K (1.9e8) | 1.5M (7.7e8) |
+------+---------+-----------+--------------+--------------+--------------+

If you get an error message during compilation about the ONNX protobuffer exceeding the maximum size,
we recommend saving weights to a file using the :code:`poptorch.Options` API :code:`options._Popart.set("saveInitializersToFile", "my_file.onnx")`.