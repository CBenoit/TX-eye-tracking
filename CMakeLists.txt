project(TX_Eye_tracker)
cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)

include_directories(include)

add_executable(TX_Eye_tracker
        src/main.cpp
        src/eye_finder.cpp
        )

target_link_libraries(TX_Eye_tracker ${OpenCV_LIBS})
