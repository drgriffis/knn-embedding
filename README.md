# k-Nearest Neighbor second-order word embeddings

Implementation of k-nearest neighborhood methods of generating second-order word embeddings, as described in:

- D Newman-Griffis and E Fosler-Lussier, "[Second-Order Word Embeddings from Nearest Neighbor Topological Features](https://arxiv.org/abs/1705.08488)." _arXiv_, arXiv:1705.08488. 2017.

This library contains two components:

1. Nearest neighbor calculation
  - Scripts: `nn_saver.py`, `nearest_neighbors.py`
  - Implemented in Tensorflow
  - Uses cosine similarity to identify nearest neighbors
2. Graph generation
  - Script: `generate_graph.py`
  - Generates a weighted, directed edgelist file compatible with [node2vec](https://github.com/aditya-grover/node2vec)

## Dependencies

A few custom libraries are included as frozen copies in the `dependencies` folder:

- `drgriffis.common.log` -- Logging utilities, from [here](https://github.com/drgriffis/miscutils)
- `pyemblib` -- Library for reading/writing word embedding files ([Github link](https://github.com/drgriffis/pyemblib))
- `configlogger` -- Library for writing runtime configuration to logfiles ([Github link](https://github.com/drgriffis/configlogger))
