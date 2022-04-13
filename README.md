# CoWalker

This is the repo for Cowalker, a high-throughput GPU random walk framework tailored for concurrent random walk queries. CoWalker introduces a multi-level concurrent execution model and a multi-dimensional scheduler to allow concurrent random walk tasks to efficiently share GPU resources with low overhead. It is able to reduce stalled GPU cores by reorganizing memory access pattern, which leads to higher throughput.
```

## Dataset
Cowalker uses [Galios](https://iss.oden.utexas.edu/?p=projects/galois) graph format (.gr) as the input. Other formats like Edgelist (form [SNAP](http://snap.stanford.edu/data/index.html)) or Matrix Market can be transformed into it with GALOIS' graph-convert tool. Compressed graphs like [Webgraph](http://law.di.unimi.it/datasets.php) need to be uncompressed first.

Here is an example:
```
wget http://snap.stanford.edu/data/wiki-Vote.txt.gz
gzip -d wiki-Vote.txt.gz
$GALOIS_PATH/build/tools/graph-convert/graph-convert -edgelist2gr  ~/data/wiki-Vote.txt  ~/data/wiki-Vote.gr
```
## Running
Please run scripts in ./scripts for testing.
# CoWalker
