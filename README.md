CUDA Shuffle
============

Implementation of GPU Shuffle using Bijective Functions

Acknowledgements
----------------

Merge Shuffle implementation used for reference comparisons:

"MergeShuffle: A Very Fast, Parallel Random Permutation Algorithm" by Bacher, Bodini, Hollender and Lumbroso (2015)
http://arxiv.org/abs/1508.03167

Implementation in `MergeShuffle.h` and `RaoSandeliusShuffle.h` is adapted from:
https://github.com/axel-bacher/mergeshuffle

WyHash Version 4 used for FeistelNetwork adapted from:
https://github.com/wangyi-fudan/wyhash