# CoWalker

This is the repo for Cowalker, a high-throughput GPU random walk framework tailored for concurrent random walk queries. Basically, it's an extention of Skywalker, our previous Alias-method-based Graph Sampling and Random Walk framework on GPUs.


## Introduction
Random walk serves as a powerful tool in dealing with large-scale graphs, reducing data size while preserving the structural information. Unfortunately, existing system frameworks all focus on the execution of a single walker task in serial. In this work, we show that conventional execution model cannot fully unleash the potential of today’s GPU cores. Simply performing coarse-grained space sharing among multiple tasks incurs unexpected performance interference. CoWalker introduces a multi-level concurrent execution model and a multi-dimensional scheduler to allow concurrent random walk tasks to efficiently share GPU resources with low overhead. It is able to reduce stalled GPU cores by reorganizing memory access pattern, which leads to higher throughput.



## Dataset
When evaluating Cowalker, we use 7 commonly used Graph datasets: web-Google, Livejournal, Orkut, Arabic-2005, UK-2005, Friendster, and SK-2005. The datasets can be downloaded from SNAP and Webgraph. You can also execute Skywalker on your preferred datasets, as long as the datasets are processed correctly as mentioned in the section of Preprosessing.

##preprocessing
Cowalker uses [Galios](https://iss.oden.utexas.edu/?p=projects/galois) graph format (.gr) as the input. Other formats like Edgelist (form [SNAP](http://snap.stanford.edu/data/index.html)) or Matrix Market can be transformed into it with GALOIS' graph-convert tool. Compressed graphs like [Webgraph](http://law.di.unimi.it/datasets.php) need to be uncompressed first.

Here is an example:
```
wget http://snap.stanford.edu/data/wiki-Vote.txt.gz
gzip -d wiki-Vote.txt.gz
$GALOIS_PATH/build/tools/graph-convert/graph-convert -edgelist2gr  ~/data/wiki-Vote.txt  ~/data/wiki-Vote.gr
```

## Running
Please run scripts in ./scripts for testing.

## Container
We have provided a dockerfile with already built executable file for easy evaluation. We created the docker container from nvidia cuda image. Cowalker is located in /home/Cowalker, while several small datasets have been placed in /home/data. To generate the complete result, users may download and preprocess the datasets from the reference in our paper. 
The container link is：
https://pan.baidu.com/s/1MRaWx_L8Gb1x4pEuTPBBEg, and the password is v2l6.
