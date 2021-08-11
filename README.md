CUDA Shuffle
============

Code and experiments for the paper [Bandwidth-Optimal Random Shuffling for GPUs](https://arxiv.org/abs/2106.06161)
```
@misc{mitchell2021bandwidthoptimal,
      title={Bandwidth-Optimal Random Shuffling for GPUs}, 
      author={Rory Mitchell and Daniel Stokes and Eibe Frank and Geoffrey Holmes},
      year={2021},
      eprint={2106.06161},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

Running the experiments
----------------
Clone this repository recursively, so that submodules are included:
```bash
git clone https://github.com/djns99/CUDA-Shuffle.git --recursive
```

This workflow has been tested on the following environment.
```
Ubuntu 18.04
CUDA 11.2
g++ 7.5
cmake 3.20.5
conda 4.10.1 with packages:
- tbb=2020.3
- matplotlib
- seaborn
```

Run the following script:
```bash
sh run_experiments.sh
```
which is expected to compile benchmarks, run the benchmarks, and present the results as figures saved in the source directory.


Acknowledgements
----------------

Merge Shuffle implementation used for reference comparisons:

"MergeShuffle: A Very Fast, Parallel Random Permutation Algorithm" by Bacher, Bodini, Hollender and Lumbroso (2015)
http://arxiv.org/abs/1508.03167

Implementation in `MergeShuffle.h` and `RaoSandeliusShuffle.h` is adapted from:
https://github.com/axel-bacher/mergeshuffle

WyHash Version 4:
https://github.com/wangyi-fudan/wyhash
