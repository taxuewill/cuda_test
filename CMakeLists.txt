# CMakeLists.txt for G4CU project

project(project)

# required cmake version
cmake_minimum_required(VERSION 2.8)


add_subdirectory(cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)


ADD_EXECUTABLE(project main.cpp)
target_link_libraries (project ${EXTRA_LIBS})

ADD_EXECUTABLE(test1 test/test.cpp)
target_link_libraries(test1 ${EXTRA_LIBS})