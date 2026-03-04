#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int root = 0;

    // -------------------------------------------------------------------------
    // 1. MPI_Bcast + MPI_Reduce to achieve the same result as MPI_Allreduce
    //    Example: every process has a value; we want the global sum on every process.
    //    Allreduce would do: everyone gets sum of all values.
    //    Emulation: Reduce to root, then Bcast from root.
    // -------------------------------------------------------------------------
    int my_value = rank + 1;  // e.g., rank 0 has 1, rank 1 has 2, ...
    int global_sum = 0;

    MPI_Reduce(&my_value, &global_sum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Bcast(&global_sum, 1, MPI_INT, root, MPI_COMM_WORLD);

    // Expected sum = 1 + 2 + ... + size = size*(size+1)/2
    int expected_sum = size * (size + 1) / 2;
    if (rank == 0) {
        std::cout << "1. Bcast+Reduce (emulating Allreduce): global_sum = " << global_sum
                  << " (expected " << expected_sum << ")\n";
    }

    // -------------------------------------------------------------------------
    // 2. MPI_Gather + MPI_Bcast to achieve the same result as MPI_Allgather
    //    Example: each process has one integer; we want everyone to have the
    //    full array of all integers.
    //    Allgather would do: everyone gets [val0, val1, ..., val_{size-1}].
    //    Emulation: Gather to root, then Bcast the full buffer from root.
    // -------------------------------------------------------------------------
    std::vector<int> my_single(1, rank);  // each rank contributes its rank
    std::vector<int> all_ranks(size);    // after emulated Allgather

    MPI_Gather(my_single.data(), 1, MPI_INT,
               all_ranks.data(), 1, MPI_INT,
               root, MPI_COMM_WORLD);
    MPI_Bcast(all_ranks.data(), size, MPI_INT, root, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "2. Gather+Bcast (emulating Allgather): all_ranks = [";
        for (int i = 0; i < size; ++i) {
            std::cout << all_ranks[i];
            if (i < size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // -------------------------------------------------------------------------
    // 3. MPI_Alltoall: personalized communication.
    //    Each process sends a distinct chunk to every other process.
    //    sendbuf[i] goes to process i; process rank receives recvbuf[j] from process j.
    //    We use one int per (sender, receiver) pair for clarity.
    // -------------------------------------------------------------------------
    std::vector<int> sendbuf(size);
    std::vector<int> recvbuf(size);

    // Rank "rank" sends to process "dest" the value (rank * 1000 + dest) as a tag
    for (int dest = 0; dest < size; ++dest) {
        sendbuf[dest] = rank * 1000 + dest;
    }

    MPI_Alltoall(sendbuf.data(), 1, MPI_INT, recvbuf.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // After Alltoall: recvbuf[src] = value that process "src" sent to us (rank)
    // So recvbuf[src] should be src * 1000 + rank
    std::cout << "3. Alltoall [rank " << rank << "]: received from each rank = [";
    for (int i = 0; i < size; ++i) {
        std::cout << recvbuf[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "] (expected [";
    for (int i = 0; i < size; ++i) {
        std::cout << (i * 1000 + rank);
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "])\n";

    MPI_Finalize();
    return 0;
}
