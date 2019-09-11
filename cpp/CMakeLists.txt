cmake_minimum_required(VERSION 3.14)
project(Neural_Networks)
find_package(Eigen3 REQUIRED)
find_package(
    PythonLibs 2.7
)
set(CMAKE_CXX_STANDARD 17)
set(HEADER_FILES 
    ${CMAKE_SOURCE_DIR}/include/Neuron.hpp
    ${CMAKE_SOURCE_DIR}/include/matplotlibcpp.h
    )
set(SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/Neuron.cpp
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    )
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    )
target_link_libraries(
    ${PROJECT_NAME}
    ${PYTHON_LIBRARIES}
)