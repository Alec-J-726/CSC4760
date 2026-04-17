// I did not test this; it is only an attempt at extra credit (grad Problem 8).
// (CSC 5760 — not required for undergrad CSC 4760.)

#include <mpi.h>

#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout
            << "HW2Problem8: MPI stub — 3-bucket quicksort / gather not implemented.\n";
    }
    MPI_Finalize();
    return 0;
}
