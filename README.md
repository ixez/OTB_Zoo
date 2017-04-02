# OTB Zoo
## Introduction
This is a collection of visual tracker's code which have been made to adapt to Visual Tracker Benchmark (OTB), making it easier to evaluate them.  
Every tracker included has a Matlab `run_XXX.m` script of OTB's framework. Trackers in C++ will be reorganized to CMake to enable compiling across multiple platforms.

- Wu, Yi, Jongwoo Lim, and Minghsuan Yang. 
"Online Object Tracking: A Benchmark." CVPR (2013).
[[paper](http://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf)]
- Wu, Yi, Jongwoo Lim, and Minghsuan Yang. 
"Object Tracking Benchmark." TPAMI (2015).
[[paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7001050&tag=1)]
[[project](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)]

## Trackers
`source` links are attached which are the original version of code. Code in our project may be modified and reorganized.  
- **MTA:** Dae Youn Lee, Jae Young Sim, Chang-Su Kim. “Multihypothesis Trajectory Analysis for Robust Visual Tracking.” CVPR (2015).
[[paper](http://mcl.korea.ac.kr/research/object_tracking/dylee_cvpr2015/dylee_cvpr_2015_paper.pdf)]
[[project](http://mcl.korea.ac.kr/research/object_tracking/dylee_cvpr2015/)]
[[source](http://mcl.korea.ac.kr/research/object_tracking/dylee_cvpr2015/dylee_cvpr_2015_source_code.zip)]
- **Struck:** S. Hare, A. Saffari, P. H. Torr. “Struck: Structured output tracking with kernels.” ICCV (2011).
[[paper](http://www.samhare.net/research/files/iccv2011_struck.pdf)]
[[project](http://www.samhare.net/research/struck)]
[[source](https://github.com/samhare/struck)]


## Others
If you need trackers' result and information, please refer to [foolwood/benchmark_results](https://github.com/foolwood/benchmark_results).