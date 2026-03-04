#include <mpi.h>
#include <iostream>

#ifndef RING_LAPS
#define RING_LAPS 1
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int TAG = 0;
    const int next = (rank + 1) % size;
    const int prev = (rank - 1 + size) % size;

    int token = 0;

    // Handle the special case of a single-process ring explicitly.
    if (size == 1) {
        for (int lap = 0; lap < RING_LAPS; ++lap) {
            // The token "goes around the ring" by staying on rank 0.
            ++token;
        }

        std::cout << "[rank 0] final token = " << token
                  << " (expected " << (RING_LAPS * size) << ")\n";

        MPI_Finalize();
        return 0;
    }

    // For size >= 2, we keep exactly one token in flight.
    // Each lap:
    //  - Rank 0 sends the token to rank 1 and then receives it from the last rank.
    //  - Every other rank receives from its predecessor, increments, and sends to its successor.
    for (int lap = 0; lap < RING_LAPS; ++lap) {
        if (rank == 0) {
            MPI_Send(&token, 1, MPI_INT, next, TAG, MPI_COMM_WORLD);

            MPI_Recv(&token, 1, MPI_INT, prev, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ++token; // increment when the token completes a lap and returns to rank 0
        } else {
            MPI_Recv(&token, 1, MPI_INT, prev, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ++token; // increment on receive at intermediate ranks
            MPI_Send(&token, 1, MPI_INT, next, TAG, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        const int expected = RING_LAPS * size;
        std::cout << "[rank 0] final token = " << token
                  << " (expected " << expected << ")\n";
    }

    MPI_Finalize();
    return 0;
}
