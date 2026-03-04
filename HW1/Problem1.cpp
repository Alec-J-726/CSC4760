#include <mpi.h>
#include <iostream>

#ifndef RING_LAPS
#define RING_LAPS 1  
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int next = (rank + 1) % size;
    const int prev = (rank - 1 + size) % size;
    const int TAG = 0;

    int token = 0;

   
    if (size == 1) {
        for (int lap = 0; lap < RING_LAPS; ++lap) {
            token++;  
        }
        std::cout << "[rank 0] final token = " << token
                  << " (expected " << (RING_LAPS * size) << ")\n";
        MPI_Finalize();
        return 0;
    }

    
    for (int lap = 0; lap < RING_LAPS; ++lap) {
        if (rank == 0) {
            MPI_Send(&token, 1, MPI_INT, next, TAG, MPI_COMM_WORLD);

            MPI_Recv(&token, 1, MPI_INT, prev, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            token++;  
        } else {
            MPI_Recv(&token, 1, MPI_INT, prev, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            token++;  
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