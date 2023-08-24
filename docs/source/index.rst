.. BESS-KGE documentation master file, created by
   sphinx-quickstart on Fri Mar 31 11:26:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BESS-KGE User Guide and API Reference
======================================

.. automodule:: besskge

   Shallow knowledge graph embedding (KGE) models are typically memory-bound, as little compute needs to be performed to score (h,r,t) triples once the embeddings of entities and relation types used in the batch have been retrieved.

   BESS (Balanced Entity Sampling and Sharing) is a KGE distribution framework designed to maximize bandwidth for gathering embeddings, by

   * storing them in fast-access IPU on-chip memory;

   * minimizing communication time for sharing embeddings between workers, leveraging balanced collective operators over high-bandwidth IPU-links.

   This allows BESS-KGE to achieve high throughput for both training and inference.

   For an introduction to the different distribution schemes used by BESS-KGE, see :ref:`BESS overview`.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   User guide <user_guide>
   BESS overview <bess>
   API reference <API_reference>
   Developers guide <dev_guide>
   Bibliography <bibliography>
