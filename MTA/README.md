## Compile
### Windows
1. CMake

Make a directory e.g. `build` in the current directory. Then  
```sh
cd build
cmake .. -DOpenCV_DIR="path_to_OpenCV" -DEigen3_INCLUDE="path_to_Eigen3"
```
`path_to_OpenCV`: Directory containing file `OpenCVConfig.cmake`.  
`path_to_Eigen3`: Directory containing directory `Eigen`.  

2.  Open `build/MTA.sln` and compile in Visual Studio.

### Linux & MacOS
```sh
mkdir build
cd build
cmake ..
make
```

## Dependencies
1. OpenCV 2.4.11
2. Eigen 3.3.3