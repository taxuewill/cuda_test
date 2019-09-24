# CMakeLists.txt for G4CU project

project(project)

# required cmake version
cmake_minimum_required(VERSION 2.8)

# add opencv package to the project
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( 
    ${OpenCV_INCLUDE_DIRS} 
    include
    ) 
SET(OpenCV_LIBRARIES ${OpenCV_LIBS})

add_subdirectory(cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)


ADD_EXECUTABLE(project main.cpp)
target_link_libraries (project ${EXTRA_LIBS})

ADD_EXECUTABLE(test1 test/test.cpp)
target_link_libraries(test1 ${EXTRA_LIBS})


add_executable(test2 test/test2.cpp )
target_link_libraries(test2 ${OpenCV_LIBRARIES} ${EXTRA_LIBS})