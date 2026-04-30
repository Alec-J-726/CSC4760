// Attempt at HW3 graduate Problem G1 (not fully validated on cluster).

#include <mpi.h>

#include <cmath>
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

    if (argc < 5) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " M N P Q\n";
            std::cerr << "Requires world_size == P*Q.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int M = std::atoi(argv[1]);  // matrix rows
    const int N = std::atoi(argv[2]);  // matrix cols
    const int P = std::atoi(argv[3]);
    const int Q = std::atoi(argv[4]);

    if (M <= 0 || N <= 0 || P <= 0 || Q <= 0 || P * Q != world_size) {
        if (world_rank == 0) {
            std::cerr << "Error: need M,N,P,Q > 0 and world_size == P*Q.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int pr = world_rank / Q;
    const int pc = world_rank % Q;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &row_comm);  // same process row
    MPI_Comm_split(MPI_COMM_WORLD, pc, pr, &col_comm);  // same process col

    // -------------------- G1 Part 1 (linear vs scatter vectors) --------------------
    // Build global vector v on rank 0, then create:
    // - linear distribution over Q columns
    // - scatter distribution over Q columns
    std::vector<double> v_global;
    if (world_rank == 0) {
        v_global.resize(static_cast<std::size_t>(N));
        for (int j = 0; j < N; ++j) v_global[static_cast<std::size_t>(j)] = 10.0 + j;
    }

    // Linear over Q columns (replicated on each process row).
    const int x_lin_n = part_count(N, Q, pc);
    const int x_lin_off = part_offset(N, Q, pc);
    std::vector<double> x_linear(static_cast<std::size_t>(x_lin_n), 0.0);
    {
        std::vector<int> cnt(static_cast<std::size_t>(Q)), dsp(static_cast<std::size_t>(Q));
        for (int c = 0; c < Q; ++c) {
            cnt[static_cast<std::size_t>(c)] = part_count(N, Q, c);
            dsp[static_cast<std::size_t>(c)] = part_offset(N, Q, c);
        }
        if (pr == 0) {
            MPI_Scatterv(pc == 0 ? v_global.data() : nullptr, cnt.data(), dsp.data(), MPI_DOUBLE,
                         x_linear.data(), x_lin_n, MPI_DOUBLE, 0, row_comm);
        }
        MPI_Bcast(x_linear.data(), x_lin_n, MPI_DOUBLE, 0, col_comm);
    }

    // Scatter/wrap over Q columns.
    const int x_scat_n = part_count(N, Q, pc);
    std::vector<double> x_scatter(static_cast<std::size_t>(x_scat_n), 0.0);
    std::vector<double> full_x(static_cast<std::size_t>(N), 0.0);
    {
        // Reconstruct full linear x on each process row, then map to scatter indices.
        std::vector<int> cnt(static_cast<std::size_t>(Q)), dsp(static_cast<std::size_t>(Q));
        for (int c = 0; c < Q; ++c) {
            cnt[static_cast<std::size_t>(c)] = part_count(N, Q, c);
            dsp[static_cast<std::size_t>(c)] = part_offset(N, Q, c);
        }
        MPI_Allgatherv(x_linear.data(), x_lin_n, MPI_DOUBLE, full_x.data(), cnt.data(), dsp.data(),
                       MPI_DOUBLE, row_comm);
        for (int j = 0; j < x_scat_n; ++j) {
            const int J = j * Q + pc;
            if (J < N) x_scatter[static_cast<std::size_t>(j)] = full_x[static_cast<std::size_t>(J)];
        }
    }

    // -------------------- G1 Part 2 (y := A*x on P x Q) --------------------
    // A is MxN stored block by row/column linear distributions:
    // rows split over P, cols split over Q.
    const int m_local = part_count(M, P, pr);
    const int n_local = part_count(N, Q, pc);
    const int m_off = part_offset(M, P, pr);
    const int n_off = part_offset(N, Q, pc);

    std::vector<double> A_local(static_cast<std::size_t>(m_local * n_local), 0.0);
    for (int i = 0; i < m_local; ++i) {
        const int I = m_off + i;
        for (int j = 0; j < n_local; ++j) {
            const int J = n_off + j;
            A_local[static_cast<std::size_t>(i * n_local + j)] = 1.0 / (1.0 + I + J);
        }
    }

    // x is already horizontal linear in x_linear; compute local partial y for owned rows.
    std::vector<double> y_partial(static_cast<std::size_t>(m_local), 0.0);
    for (int i = 0; i < m_local; ++i) {
        double s = 0.0;
        for (int j = 0; j < n_local; ++j) {
            s += A_local[static_cast<std::size_t>(i * n_local + j)] *
                 x_linear[static_cast<std::size_t>(j)];
        }
        y_partial[static_cast<std::size_t>(i)] = s;
    }

    // Sum contributions across columns so each row has final y for its local rows.
    std::vector<double> y_rows(static_cast<std::size_t>(m_local), 0.0);
    MPI_Allreduce(y_partial.data(), y_rows.data(), m_local, MPI_DOUBLE, MPI_SUM, row_comm);

    // y distribution: vertical (linear over P rows), then replicated across columns.
    MPI_Bcast(y_rows.data(), m_local, MPI_DOUBLE, 0, row_comm);

    // Gather y on rank 0 for quick check.
    std::vector<double> y_global;
    if (world_rank == 0) y_global.resize(static_cast<std::size_t>(M), 0.0);
    {
        std::vector<int> cnt(static_cast<std::size_t>(P)), dsp(static_cast<std::size_t>(P));
        for (int r = 0; r < P; ++r) {
            cnt[static_cast<std::size_t>(r)] = part_count(M, P, r);
            dsp[static_cast<std::size_t>(r)] = part_offset(M, P, r);
        }
        if (pc == 0) {
            MPI_Gatherv(y_rows.data(), m_local, MPI_DOUBLE,
                        world_rank == 0 ? y_global.data() : nullptr, cnt.data(), dsp.data(),
                        MPI_DOUBLE, 0, col_comm);
        }
    }

    // -------------------- G1 extra credit attempt: z := A*y (assume M==N) --------------------
    if (M == N) {
        // Need y horizontal over Q to multiply by A again.
        std::vector<double> y_full(static_cast<std::size_t>(M), 0.0);
        {
            std::vector<int> cnt(static_cast<std::size_t>(P)), dsp(static_cast<std::size_t>(P));
            for (int r = 0; r < P; ++r) {
                cnt[static_cast<std::size_t>(r)] = part_count(M, P, r);
                dsp[static_cast<std::size_t>(r)] = part_offset(M, P, r);
            }
            MPI_Allgatherv(y_rows.data(), m_local, MPI_DOUBLE, y_full.data(), cnt.data(), dsp.data(),
                           MPI_DOUBLE, col_comm);
        }
        std::vector<double> y_horiz_local(static_cast<std::size_t>(n_local), 0.0);
        for (int j = 0; j < n_local; ++j) {
            y_horiz_local[static_cast<std::size_t>(j)] = y_full[static_cast<std::size_t>(n_off + j)];
        }

        std::vector<double> z_partial(static_cast<std::size_t>(m_local), 0.0);
        for (int i = 0; i < m_local; ++i) {
            double s = 0.0;
            for (int j = 0; j < n_local; ++j) {
                s += A_local[static_cast<std::size_t>(i * n_local + j)] *
                     y_horiz_local[static_cast<std::size_t>(j)];
            }
            z_partial[static_cast<std::size_t>(i)] = s;
        }
        std::vector<double> z_rows(static_cast<std::size_t>(m_local), 0.0);
        MPI_Allreduce(z_partial.data(), z_rows.data(), m_local, MPI_DOUBLE, MPI_SUM, row_comm);

        // redistribute z like x (horizontal over Q): first collect full z by rows then scatter on row 0.
        std::vector<double> z_full(static_cast<std::size_t>(M), 0.0);
        {
            std::vector<int> cnt(static_cast<std::size_t>(P)), dsp(static_cast<std::size_t>(P));
            for (int r = 0; r < P; ++r) {
                cnt[static_cast<std::size_t>(r)] = part_count(M, P, r);
                dsp[static_cast<std::size_t>(r)] = part_offset(M, P, r);
            }
            MPI_Allgatherv(z_rows.data(), m_local, MPI_DOUBLE, z_full.data(), cnt.data(), dsp.data(),
                           MPI_DOUBLE, col_comm);
        }
        std::vector<double> z_horiz_local(static_cast<std::size_t>(n_local), 0.0);
        for (int j = 0; j < n_local; ++j) {
            z_horiz_local[static_cast<std::size_t>(j)] = z_full[static_cast<std::size_t>(n_off + j)];
        }
        if (world_rank == 0) {
            std::cout << "G1 extra-credit attempt computed z := A*y (M==N case), "
                         "stored horizontally over Q.\n";
        }
    }

    if (world_rank == 0) {
        std::cout << "G1 part1 done: built linear and scatter vector forms over Q columns.\n";
        std::cout << "G1 part2 done: computed y := A*x on P x Q grid.\n";
        std::cout << "Distribution note: y is vertical (split by rows over P), "
                     "and replicated across Q columns.\n";
        if (!y_global.empty()) {
            std::cout << "y[0..min(4,M-1)] = ";
            for (int i = 0; i < M && i < 5; ++i) {
                std::cout << y_global[static_cast<std::size_t>(i)] << (i + 1 < M && i < 4 ? ' ' : '\n');
            }
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
