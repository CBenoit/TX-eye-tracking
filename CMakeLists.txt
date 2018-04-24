cmake_minimum_required(VERSION 3.9)
project(TX_eye_tracking)

set(CMAKE_CXX_STANDARD 17)

add_compile_options("-DNDEBUG")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(TX_eye_tracking main.cpp)
target_link_libraries(TX_eye_tracking ${OpenCV_LIBS} )
