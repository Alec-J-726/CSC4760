#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Define P and Q such that P * Q = world_size.
    // Here we explicitly choose Q; ensure world_size == P * Q when launching.
    int Q = 2;
    int P = world_size / Q;

    // Requirement: World size must be exactly P * Q
    if (world_size != P * Q) {
        if (world_rank == 0) {
            std::cerr << "Error: World size must be a multiple of Q=" << Q << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 1. First split: same color when ranks are divided by Q (integer division).
    // This effectively groups processes into "rows".
    int row_color = world_rank / Q;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, world_rank, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    // 2. Second split: same color when rank is rank mod Q.
    // This effectively groups processes into "columns".
    int col_color = world_rank % Q;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, world_rank, &col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // Demonstration of the groups
    printf("World Rank %d: [Row Comm] color %d, rank %d/%d | [Col Comm] color %d, rank %d/%d\n",
           world_rank, row_color, row_rank, row_size, col_color, col_rank, col_size);

    // Clean up
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();

    return 0;
}