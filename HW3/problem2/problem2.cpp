#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

static int part_count(int N, int K, int k) {
    const int base = N / K;
    const int rem = N % K;
    return base + (k < rem ? 1 : 0);
}

static int part_offset(int N, int K, int k) {
    const int base = N / K;
    const int rem = N % K;
    return k * base + (k < rem ? k : rem);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " M P Q\n";
            std::cerr << "Requires world_size == P*Q.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int M = std::atoi(argv[1]);
    const int P = std::atoi(argv[2]);
    const int Q = std::atoi(argv[3]);

    if (M < 0 || P < 1 || Q < 1) {
        if (world_rank == 0) {
            std::cerr << "Error: require M >= 0, P >= 1, Q >= 1.\n";
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

    // 2D coordinates in a row-major P x Q grid.
    const int row = world_rank / Q;  // [0, P)
    const int col = world_rank % Q;  // [0, Q)

    // row_comm: fixed row, varying col (horizontal)
    // col_comm: fixed col, varying row (vertical)
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    const int x_local_n = part_count(M, P, row);      // vertical distribution over rows
    const int x_local_off = part_offset(M, P, row);
    const int y_local_n = part_count(M, Q, col);      // horizontal distribution over cols
    const int y_local_off = part_offset(M, Q, col);

    std::vector<int> x_local(static_cast<std::size_t>(x_local_n), 0);
    std::vector<int> y_local(static_cast<std::size_t>(y_local_n), -1);

    // Create test vector x only on process (0,0).
    std::vector<int> x_global_root;
    if (row == 0 && col == 0) {
        x_global_root.resize(static_cast<std::size_t>(M));
        for (int j = 0; j < M; ++j) {
            x_global_root[static_cast<std::size_t>(j)] = 1000 + j;
        }
    }

    // Scatter x down the first column only.
    if (col == 0) {
        std::vector<int> counts_p(static_cast<std::size_t>(P));
        std::vector<int> displs_p(static_cast<std::size_t>(P));
        for (int r = 0; r < P; ++r) {
            counts_p[static_cast<std::size_t>(r)] = part_count(M, P, r);
            displs_p[static_cast<std::size_t>(r)] = part_offset(M, P, r);
        }

        MPI_Scatterv(
            row == 0 ? x_global_root.data() : nullptr, counts_p.data(), displs_p.data(), MPI_INT,
            x_local.data(), x_local_n, MPI_INT, 0, col_comm);
    }

    // Broadcast each row's x chunk from column 0 across that row.
    MPI_Bcast(x_local.data(), x_local_n, MPI_INT, 0, row_comm);

    // Build full x on each process by gathering row chunks along its column.
    // Since x is replicated across columns after row-wise broadcast, every column can do this.
    std::vector<int> counts_p(static_cast<std::size_t>(P));
    std::vector<int> displs_p(static_cast<std::size_t>(P));
    for (int r = 0; r < P; ++r) {
        counts_p[static_cast<std::size_t>(r)] = part_count(M, P, r);
        displs_p[static_cast<std::size_t>(r)] = part_offset(M, P, r);
    }
    std::vector<int> x_full(static_cast<std::size_t>(M), 0);
    MPI_Allgatherv(x_local.data(), x_local_n, MPI_INT, x_full.data(), counts_p.data(),
                   displs_p.data(), MPI_INT, col_comm);

    // Parallel copy y := x, with y stored horizontally (distributed over Q by column).
    for (int j = 0; j < y_local_n; ++j) {
        y_local[static_cast<std::size_t>(j)] =
            x_full[static_cast<std::size_t>(y_local_off + j)];
    }

    // Verify: gather y in each row and compare with expected x.
    std::vector<int> counts_q(static_cast<std::size_t>(Q));
    std::vector<int> displs_q(static_cast<std::size_t>(Q));
    for (int c = 0; c < Q; ++c) {
        counts_q[static_cast<std::size_t>(c)] = part_count(M, Q, c);
        displs_q[static_cast<std::size_t>(c)] = part_offset(M, Q, c);
    }

    std::vector<int> y_row_full;
    if (col == 0) {
        y_row_full.resize(static_cast<std::size_t>(M), 0);
    }
    MPI_Gatherv(y_local.data(), y_local_n, MPI_INT, col == 0 ? y_row_full.data() : nullptr,
                counts_q.data(), displs_q.data(), MPI_INT, 0, row_comm);

    if (col == 0) {
        bool ok = true;
        for (int j = 0; j < M; ++j) {
            const int expected = 1000 + j;
            if (y_row_full[static_cast<std::size_t>(j)] != expected) {
                ok = false;
                break;
            }
        }
        std::cout << "[row " << row << " root] y replica check: "
                  << (ok ? "PASS" : "FAIL") << "\n";
    }

    if (world_rank == 0) {
        std::cout << "Problem2 complete: built P x Q topology (" << P << " x " << Q
                  << "), scattered x on column 0, broadcast by rows, and copied y := x.\n";
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
