# Struck

## Compatibility
Windows :white_check_mark: Linux :question: MacOS :white_check_mark:  

## Compile
### Windows
1. CMake

Make a directory e.g. `build` in the current directory. Then  
```sh
cd build
cmake .. -DOpenCV_DIR="path_to_OpenCV" -DEIGEN3_INCLUDE_DIRS="path_to_Eigen3"
```
`path_to_OpenCV`: Directory containing the file `OpenCVConfig.cmake`.  
`path_to_Eigen3`: Directory containing the directory `Eigen`.  

2. Open `build/Struck.sln` and compile in Visual Studio.

### Linux & MacOS
```sh
mkdir build
cd build
cmake ..
make
```

## Usage
Copy Matlab script `run_Struck.m`, configuration file `config.txt` and compiled Struck excutable file into OTB's tracker directory.

## Dependencies
1. OpenCV 2.4.11
2. Eigen 3.3.3