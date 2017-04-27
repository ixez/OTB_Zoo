# OTB Zoo
## Introduction
This is a collection of visual trackers' code which have been made to adapt to Visual Tracker Benchmark (OTB), making it easier to evaluate them.  
Every tracker included has a Matlab `run_XXX.m` script of OTB's framework. Trackers in C++ will be reorganized to CMake to enable compiling across multiple platforms.

- Wu, Yi, Jongwoo Lim, and Minghsuan Yang. 
"Online Object Tracking: A Benchmark." CVPR (2013).
[[paper](http://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf)]
- Wu, Yi, Jongwoo Lim, and Minghsuan Yang. 
"Object Tracking Benchmark." TPAMI (2015).
[[paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7001050&tag=1)]
[[project](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)]

## Trackers
[source](#) links are attached which are the original version of code. Codes in our project may be modified and reorganized.  
- **KCF:** João F. Henriques, Rui Caseiro, Pedro Martins, Jorge Batista. 
"High-Speed Tracking with Kernelized Correlation Filters." TPAMI (2015).
[[paper](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)]
[[project](http://www.robots.ox.ac.uk/~joao/circulant/)]
[[source](https://github.com/joaofaro/KCFcpp)]

- **MTA:** Dae Youn Lee, Jae Young Sim, Chang-Su Kim. “Multihypothesis Trajectory Analysis for Robust Visual Tracking.” CVPR (2015).
[[paper](http://mcl.korea.ac.kr/research/object_tracking/dylee_cvpr2015/dylee_cvpr_2015_paper.pdf)]
[[project](http://mcl.korea.ac.kr/research/object_tracking/dylee_cvpr2015/)]
[[source](http://mcl.korea.ac.kr/research/object_tracking/dylee_cvpr2015/dylee_cvpr_2015_source_code.zip)]

- **Struck:** S. Hare, A. Saffari, P. H. Torr. “Struck: Structured output tracking with kernels.” ICCV (2011).
[[paper](http://www.samhare.net/research/files/iccv2011_struck.pdf)]
[[project](http://www.samhare.net/research/struck)]
[[source](https://github.com/samhare/struck)]

## Notes
- CMake of version higher than 3.0 is preferable. Lower one could fail in some trackers which requires C++11.
- Compiling into 64bit is preferable. Newer version of OpenCV and some other dependencies might require 64bit.
- If you are using Visual Studio to compile, please remeber to switch to `Release` mode for computing efficiency.

## Remarks
- If you need trackers' result and information, please refer to [foolwood/benchmark_results](https://github.com/foolwood/benchmark_results).

## 介绍
这是一个收集和整理视觉跟踪器代码的项目，跟踪器代码被微调以适应OTB跟踪器评估标准，配备了对应的`run_XXX.m`的Matlab脚本，并且将所有C++项目以CMake进行重新组织，使之能够在不同平台下进行运行。

## 注意事项
- 最好使用CMake 3.0以上的版本，部分跟踪器要求C++11，可能会导致低版本CMake错误。
- 最好以64位进行编译，OpenCV和一些依赖可能要求必须使用64位。
- 如果你使用Visual Studio进行编译，请记得切换到Release模式编译，以获得更快的运行速度。