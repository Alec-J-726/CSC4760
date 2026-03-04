#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int N = 1024; 
    
    
    int local_n = N / size;
    int remainder = N % size;
    
    if (rank < remainder) local_n++;

    
    std::vector<double> vecA(local_n, 1.0);
    std::vector<double> vecB(local_n, 2.0);

    
    double start_time = MPI_Wtime();

    
    double local_dot = 0.0;
    for (int i = 0; i < local_n; ++i) {
        local_dot += vecA[i] * vecB[i];
    }

    
    int steps = ceil(log2(size));
    for (int i = 0; i < steps; i++) {
        int partner_distance = pow(2, i);
        
        if (rank % (2 * partner_distance) == 0) {
            int source = rank + partner_distance;
            if (source < size) {
                double received_sum;
                MPI_Recv(&received_sum, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_dot += received_sum;
            }
        } else if (rank % (2 * partner_distance) == partner_distance) {
            int dest = rank - partner_distance;
            MPI_Send(&local_dot, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    }

    double end_time = MPI_Wtime();

    
    if (rank == 0) {
        std::cout << "N: " << N << " | P: " << size << std::endl;
        std::cout << "Result: " << local_dot << " (Expected: " << N * 2.0 << ")" << std::endl;
        std::cout << "Time: " << (end_time - start_time) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}