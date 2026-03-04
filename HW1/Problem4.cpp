#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Problem size N can be varied at compile time or via argument;
    // here we keep it simple and hard-code one value like the LLM might.
    int N = 1024;

    // Divide work as evenly as possible: first (N % size) ranks get one extra.
    int base = N / size;
    int remainder = N % size;
    int local_n = base + (rank < remainder ? 1 : 0);

    // Use easy-to-check data that isn't all zeros.
    // For simplicity, the LLM initializes local chunks independently.
    std::vector<double> a(local_n, 1.0);
    std::vector<double> b(local_n, 2.0);

    double t0 = MPI_Wtime();

    // Local dot product over the assigned chunk.
    double local_dot = 0.0;
    for (int i = 0; i < local_n; ++i) {
        local_dot += a[i] * b[i];
    }

    // Manual tree reduction using only point-to-point sends/receives.
    // This pattern assumes no collectives like MPI_Reduce.
    int max_steps = static_cast<int>(std::ceil(std::log2(static_cast<double>(size))));
    for (int step = 0; step < max_steps; ++step) {
        int offset = 1 << step;  // 2^step
        int pair_span = offset << 1;

        if (rank % pair_span == 0) {
            int src = rank + offset;
            if (src < size) {
                double recv_val = 0.0;
                MPI_Recv(&recv_val, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_dot += recv_val;
            }
        } else if (rank % pair_span == offset) {
            int dest = rank - offset;
            MPI_Send(&local_dot, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            // After sending, this rank is done contributing to the reduction.
            break;
        }
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {
        // In this simple initialization, each local element contributes 1.0 * 2.0 = 2.0.
        // If N is exactly the sum of all local_n over ranks, expected = 2.0 * N.
        std::cout << "N = " << N << ", P = " << size << "\n";
        std::cout << "Dot product = " << local_dot
                  << " (expected approximately " << 2.0 * N << ")\n";
        std::cout << "Elapsed time = " << (t1 - t0) << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}

