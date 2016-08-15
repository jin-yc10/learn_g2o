cmake_minimum_required(VERSION 3.2)
project(optimize)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -MMD -MP -pg")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake_modules)

set(G2O_ROOT ${CMAKE_SOURCE_DIR}/)

find_package(OpenCV REQUIRED)
FIND_PACKAGE(Cholmod)
FIND_PACKAGE(BLAS)
FIND_PACKAGE(LAPACK)
find_package(g2o)

if (G2O_SOLVERS_FOUND)
    message(INFO ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src
        ${CHOLMOD_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        ${G2O_INCLUDE_DIR}
        /usr/include/eigen3/)
link_directories(/usr/lib/x86_64-linux-gnu/)

add_executable(test_optimize main.cpp)
target_link_libraries(test_optimize ${OpenCV_LIBS}
        ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${CHOLMOD_LIBRARIES}
        gflags glog)

add_executable(bal_example bal_example.cpp)
target_link_libraries(bal_example ${OpenCV_LIBS}
        ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${CHOLMOD_LIBRARIES}
        gflags glog)