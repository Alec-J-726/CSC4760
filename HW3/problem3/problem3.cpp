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

    const int row = world_rank / Q;  // [0, P)
    const int col = world_rank % Q;  // [0, Q)

    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    // x is linear vertical over P rows (replicated over Q columns after row broadcast).
    const int x_local_n = part_count(M, P, row);
    std::vector<int> x_local(static_cast<std::size_t>(x_local_n), 0);

    // y is scatter/wrap over Q columns in each row:
    // global J on row maps to (col = J % Q, local j = J / Q).
    const int y_local_n = part_count(M, Q, col);
    std::vector<int> y_local(static_cast<std::size_t>(y_local_n), -1);

    // Root data only at process (0,0).
    std::vector<int> x_root;
    if (row == 0 && col == 0) {
        x_root.resize(static_cast<std::size_t>(M));
        for (int j = 0; j < M; ++j) {
            x_root[static_cast<std::size_t>(j)] = 1000 + j;
        }
    }

    // Scatter x down first column (linear over P).
    if (col == 0) {
        std::vector<int> counts_p(static_cast<std::size_t>(P));
        std::vector<int> displs_p(static_cast<std::size_t>(P));
        for (int r = 0; r < P; ++r) {
            counts_p[static_cast<std::size_t>(r)] = part_count(M, P, r);
            displs_p[static_cast<std::size_t>(r)] = part_offset(M, P, r);
        }
        MPI_Scatterv(row == 0 ? x_root.data() : nullptr, counts_p.data(), displs_p.data(),
                     MPI_INT, x_local.data(), x_local_n, MPI_INT, 0, col_comm);
    }

    // Broadcast each row chunk from column 0 across columns.
    MPI_Bcast(x_local.data(), x_local_n, MPI_INT, 0, row_comm);

    // Reconstruct full x along each column so each process can compute wrapped y pieces.
    std::vector<int> counts_p(static_cast<std::size_t>(P));
    std::vector<int> displs_p(static_cast<std::size_t>(P));
    for (int r = 0; r < P; ++r) {
        counts_p[static_cast<std::size_t>(r)] = part_count(M, P, r);
        displs_p[static_cast<std::size_t>(r)] = part_offset(M, P, r);
    }
    std::vector<int> x_full(static_cast<std::size_t>(M), 0);
    MPI_Allgatherv(x_local.data(), x_local_n, MPI_INT, x_full.data(), counts_p.data(),
                   displs_p.data(), MPI_INT, col_comm);

    // Compute y := x with y in scatter distribution over Q columns.
    // Local y index j corresponds to global J = j*Q + col.
    for (int j = 0; j < y_local_n; ++j) {
        const int J = j * Q + col;
        if (J < M) {
            y_local[static_cast<std::size_t>(j)] = x_full[static_cast<std::size_t>(J)];
        }
    }

    // Verify each row replica by gathering wrapped y from columns back to row root.
    std::vector<int> y_row_full;
    if (col == 0) {
        y_row_full.assign(static_cast<std::size_t>(M), -1);
    }
    for (int c = 0; c < Q; ++c) {
        int recv_count = part_count(M, Q, c);
        std::vector<int> tmp(static_cast<std::size_t>(recv_count), 0);
        if (col == c) {
            tmp = y_local;
        }
        MPI_Bcast(tmp.data(), recv_count, MPI_INT, c, row_comm);
        if (col == 0) {
            for (int j = 0; j < recv_count; ++j) {
                const int J = j * Q + c;
                if (J < M) {
                    y_row_full[static_cast<std::size_t>(J)] = tmp[static_cast<std::size_t>(j)];
                }
            }
        }
    }

    if (col == 0) {
        bool ok = true;
        for (int J = 0; J < M; ++J) {
            const int expected = 1000 + J;
            if (y_row_full[static_cast<std::size_t>(J)] != expected) {
                ok = false;
                break;
            }
        }
        std::cout << "[row " << row << " root] wrapped y replica check: "
                  << (ok ? "PASS" : "FAIL") << "\n";
    }

    if (world_rank == 0) {
        std::cout << "Problem3 complete: y stored in scatter (wrap) distribution over Q columns.\n";
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
