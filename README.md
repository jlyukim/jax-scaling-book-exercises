# jax-scaling-book-exercises
Worked code exercises from self-study of [JAX Scaling Book](https://jax-ml.github.io/scaling-book/).

## Covers: 
- Model rooflines & arithmetic intensity
- Basic collectives: AllGather, AllReduce, ReduceScatter, AllToAll
- JAX code
  - Device meshes
  - PartitionSpec
  - Benchmarking different collectives 
  - Profiling & inspecting HLO

GCP/Kaggle no longer offer free multi-TPU access -> experiments done on virtual CPU devices, expect inconsistencies in timing 
