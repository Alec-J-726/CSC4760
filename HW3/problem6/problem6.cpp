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
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);  // same row
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);  // same col

    // x: initially horizontal linear over Q columns (same in each row once distributed).
    const int x_local_n = part_count(M, Q, col);
    const int x_local_off = part_offset(M, Q, col);
    std::vector<double> x_local(static_cast<std::size_t>(x_local_n), 0.0);

    // y: initially vertical linear over P rows (same in each column once distributed).
    const int y_local_n = part_count(M, P, row);
    const int y_local_off = part_offset(M, P, row);
    std::vector<double> y_local(static_cast<std::size_t>(y_local_n), 0.0);

    // Root vectors at process (0,0) only.
    std::vector<double> x_root;
    std::vector<double> y_root;
    if (row == 0 && col == 0) {
        x_root.resize(static_cast<std::size_t>(M));
        y_root.resize(static_cast<std::size_t>(M));
        for (int j = 0; j < M; ++j) {
            x_root[static_cast<std::size_t>(j)] = 1.0 + j;           // 1,2,3,...
            y_root[static_cast<std::size_t>(j)] = 2.0 * (1.0 + j);   // 2,4,6,...
        }
    }

    // Distribute x on row 0 across columns, then broadcast down columns.
    if (row == 0) {
        std::vector<int> counts_q(static_cast<std::size_t>(Q));
        std::vector<int> displs_q(static_cast<std::size_t>(Q));
        for (int c = 0; c < Q; ++c) {
            counts_q[static_cast<std::size_t>(c)] = part_count(M, Q, c);
            displs_q[static_cast<std::size_t>(c)] = part_offset(M, Q, c);
        }
        MPI_Scatterv(col == 0 ? x_root.data() : nullptr, counts_q.data(), displs_q.data(),
                     MPI_DOUBLE, x_local.data(), x_local_n, MPI_DOUBLE, 0, row_comm);
    }
    MPI_Bcast(x_local.data(), x_local_n, MPI_DOUBLE, 0, col_comm);

    // Distribute y on column 0 across rows, then broadcast across rows.
    if (col == 0) {
        std::vector<int> counts_p(static_cast<std::size_t>(P));
        std::vector<int> displs_p(static_cast<std::size_t>(P));
        for (int r = 0; r < P; ++r) {
            counts_p[static_cast<std::size_t>(r)] = part_count(M, P, r);
            displs_p[static_cast<std::size_t>(r)] = part_offset(M, P, r);
        }
        MPI_Scatterv(row == 0 ? y_root.data() : nullptr, counts_p.data(), displs_p.data(),
                     MPI_DOUBLE, y_local.data(), y_local_n, MPI_DOUBLE, 0, col_comm);
    }
    MPI_Bcast(y_local.data(), y_local_n, MPI_DOUBLE, 0, row_comm);

    // Build full x and full y everywhere to align indices and compute local partial dot.
    std::vector<int> counts_q(static_cast<std::size_t>(Q));
    std::vector<int> displs_q(static_cast<std::size_t>(Q));
    for (int c = 0; c < Q; ++c) {
        counts_q[static_cast<std::size_t>(c)] = part_count(M, Q, c);
        displs_q[static_cast<std::size_t>(c)] = part_offset(M, Q, c);
    }
    std::vector<double> x_full(static_cast<std::size_t>(M), 0.0);
    MPI_Allgatherv(x_local.data(), x_local_n, MPI_DOUBLE, x_full.data(), counts_q.data(),
                   displs_q.data(), MPI_DOUBLE, row_comm);

    std::vector<int> counts_p(static_cast<std::size_t>(P));
    std::vector<int> displs_p(static_cast<std::size_t>(P));
    for (int r = 0; r < P; ++r) {
        counts_p[static_cast<std::size_t>(r)] = part_count(M, P, r);
        displs_p[static_cast<std::size_t>(r)] = part_offset(M, P, r);
    }
    std::vector<double> y_full(static_cast<std::size_t>(M), 0.0);
    MPI_Allgatherv(y_local.data(), y_local_n, MPI_DOUBLE, y_full.data(), counts_p.data(),
                   displs_p.data(), MPI_DOUBLE, col_comm);

    // To avoid counting all M terms on every rank, assign each global index J to exactly one rank:
    // owner row = J % P, owner col = J % Q.
    double local_dot = 0.0;
    for (int J = 0; J < M; ++J) {
        if ((J % P) == row && (J % Q) == col) {
            local_dot += x_full[static_cast<std::size_t>(J)] * y_full[static_cast<std::size_t>(J)];
        }
    }

    double dot_all = 0.0;
    MPI_Allreduce(&local_dot, &dot_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Expected for x_j=(j+1), y_j=2*(j+1): dot = 2 * sum_{k=1..M} k^2
    double expected = 0.0;
    for (int k = 1; k <= M; ++k) {
        expected += 2.0 * static_cast<double>(k) * static_cast<double>(k);
    }

    if (world_rank == 0) {
        std::cout << "Dot product result (replicated on all ranks): " << dot_all << "\n";
        std::cout << "Expected value: " << expected << "\n";
        std::cout << "Abs error: " << (dot_all > expected ? dot_all - expected : expected - dot_all)
                  << "\n";
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
