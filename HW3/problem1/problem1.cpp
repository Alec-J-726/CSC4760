#include <mpi.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 3) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " P Q\n";
            std::cerr << "Requires world_size == P*Q.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int P = std::atoi(argv[1]);
    const int Q = std::atoi(argv[2]);

    if (P < 1 || Q < 1) {
        if (world_rank == 0) {
            std::cerr << "Error: P and Q must both be >= 1.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (P * Q != world_size) {
        if (world_rank == 0) {
            std::cerr << "Error: world_size=" << world_size
                      << " but expected P*Q=" << (P * Q) << ".\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Split #1: color = rank / Q. Each group has Q ranks.
    const int color1 = world_rank / Q;
    const int key1 = world_rank;
    MPI_Comm comm1;
    MPI_Comm_split(MPI_COMM_WORLD, color1, key1, &comm1);

    int comm1_rank = 0;
    int comm1_size = 0;
    MPI_Comm_rank(comm1, &comm1_rank);
    MPI_Comm_size(comm1, &comm1_size);

    int sum_ranks = 0;
    MPI_Reduce(&world_rank, &sum_ranks, 1, MPI_INT, MPI_SUM, 0, comm1);

    if (comm1_rank == 0) {
        std::cout << "[Split1 root] color=" << color1 << " size=" << comm1_size
                  << " sum(world_ranks)=" << sum_ranks << "\n";
    }

    // Split #2: color = rank % Q. Each group has P ranks.
    const int color2 = world_rank % Q;
    const int key2 = world_rank;
    MPI_Comm comm2;
    MPI_Comm_split(MPI_COMM_WORLD, color2, key2, &comm2);

    int comm2_rank = 0;
    int comm2_size = 0;
    MPI_Comm_rank(comm2, &comm2_rank);
    MPI_Comm_size(comm2, &comm2_size);

    int root_world_rank = world_rank;
    if (comm2_rank == 0) {
        root_world_rank = world_rank;
    }
    MPI_Bcast(&root_world_rank, 1, MPI_INT, 0, comm2);

    // Print in world-rank order so output is easier to read.
    for (int r = 0; r < world_size; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == r) {
            std::cout << "[Split2 recv] world_rank=" << world_rank
                      << " color=" << color2
                      << " group_size=" << comm2_size
                      << " received_root_world_rank=" << root_world_rank << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_free(&comm1);
    MPI_Comm_free(&comm2);
    MPI_Finalize();
    return 0;
}
