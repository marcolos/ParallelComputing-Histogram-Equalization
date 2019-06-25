#[[  CMAKE PER COMPILATORE ESTERNO GCC-8 DI BREW (NON FUNZIONA)
cmake_minimum_required(VERSION 3.13)
project(OpenMP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#AGGIUNTO IO ###init
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-8")
set(CMAKE_CXX_FLAGS "-fopenmp")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
###end

add_executable(OpenMP main.cpp)
target_link_libraries(OpenMP ${OpenCV_LIBS})
]]


#[[ CON LLVM
cmake_minimum_required(VERSION 3.10)
project( OpenMP )

link_directories("/usr/local/Cellar/llvm/8.0.0/lib")
add_executable( OpenMP main.cpp )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( OpenMP ${OpenCV_LIBS} )

set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/8.0.0/bin/clang++")
# try also to compile and execute without: -fopenmp
set(CMAKE_CXX_FLAGS "-I/usr/local/Cellar/llvm/8.0.0/include -fopenmp")


set(SOURCE_FILES main.cpp)
]]


cmake_minimum_required(VERSION 3.3)
project(OpenMP)

set(CMAKE_CXX_STANDARD 11)

# try also to compile and execute without: -fopenmp
# in Linux just use -fopenmp, without -lomp and -Xpreprocessor
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp")
# the following 2 lines are needed on macOS
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# le seguenti 2 righe per inserire openCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#[[include_directories(/usr/local/Cellar/opencv/4.0.1/include/opencv4/opencv)
link_directories(/usr/local/Cellar/opencv/4.0.1/lib)]]


set(SOURCE_FILES main.cpp)
add_executable(OpenMP ${SOURCE_FILES})

#riga necessaria per OpenCV
target_link_libraries( OpenMP ${OpenCV_LIBS} )