# HW2 Kokkos problems — build after: spack load kokkos ...
#   cd HW2
#   cmake -S . -B build
#   cmake --build build
#   ./build/HW2Problem1   (requires MPI)
#   ./build/HW2Problem2
#   ./build/HW2Problem3
#   ./build/HW2Problem4
#   ./build/HW2Problem5
#   ./build/HW2Problem6
#   ./build/HW2Problem7
#   ./build/HW2Problem8   (requires MPI)
#   ./build/HW2Problem9
#   ./build/HW2Problem10

cmake_minimum_required(VERSION 3.16)
project(HW2_Kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(HW2Problem2 HW2Problem2.cpp)
target_link_libraries(HW2Problem2 PRIVATE Kokkos::kokkos)

add_executable(HW2Problem3 HW2Problem3.cpp)
target_link_libraries(HW2Problem3 PRIVATE Kokkos::kokkos)

add_executable(HW2Problem4 HW2Problem4.cpp)
target_link_libraries(HW2Problem4 PRIVATE Kokkos::kokkos)

add_executable(HW2Problem5 HW2Problem5.cpp)
target_link_libraries(HW2Problem5 PRIVATE Kokkos::kokkos)

add_executable(HW2Problem6 HW2Problem6.cpp)
target_link_libraries(HW2Problem6 PRIVATE Kokkos::kokkos)

add_executable(HW2Problem7 HW2Problem7.cpp)
target_link_libraries(HW2Problem7 PRIVATE Kokkos::kokkos)

find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
    # If `mpirun -n 4` still prints P=1, CMake linked a different MPI than your launcher.
    # Reconfigure from a clean build dir after `spack load openmpi`, using the same
    # wrapper mpirun uses, e.g.:
    #   rm -rf build && cmake -S . -B build -DCMAKE_CXX_COMPILER=$(which mpicxx) && cmake --build build
    message(STATUS "MPI CXX compiler (for HW2Problem1 link): ${MPI_CXX_COMPILER}")
    add_executable(HW2Problem1 HW2Problem1.cpp)
    target_link_libraries(HW2Problem1 PRIVATE MPI::MPI_CXX)
    add_executable(HW2Problem8 HW2Problem8.cpp)
    target_link_libraries(HW2Problem8 PRIVATE MPI::MPI_CXX)
else()
    message(STATUS "MPI not found: skipping HW2Problem1 and HW2Problem8 (MPI)")
endif()

add_executable(HW2Problem9 HW2Problem9.cpp)
target_link_libraries(HW2Problem9 PRIVATE Kokkos::kokkos)

add_executable(HW2Problem10 HW2Problem10.cpp)
target_link_libraries(HW2Problem10 PRIVATE Kokkos::kokkos)
