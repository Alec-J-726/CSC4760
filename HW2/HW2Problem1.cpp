#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

static void merge_sorted_into(const std::vector<int>& a, const std::vector<int>& b,
                              std::vector<int>& out) {
    out.resize(a.size() + b.size());
    std::size_t i = 0, j = 0, k = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j]) {
            out[k++] = a[i++];
        } else {
            out[k++] = b[j++];
        }
    }
    while (i < a.size()) {
        out[k++] = a[i++];
    }
    while (j < b.size()) {
        out[k++] = b[j++];
    }
}

static void merge_sort(std::vector<int>& v) {
    const std::size_t n = v.size();
    if (n <= 1) {
        return;
    }
    const std::size_t mid = n / 2;
    std::vector<int> left(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    std::vector<int> right(v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    merge_sort(left);
    merge_sort(right);
    merge_sorted_into(left, right, v);
}

static bool is_sorted(const std::vector<int>& v) {
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i - 1] > v[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int N = 256;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    if (N < 0) {
        if (rank == 0) {
            std::cerr << "N must be non-negative.\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<int> global;
    if (rank == 0) {
        global.resize(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
            global[static_cast<std::size_t>(i)] = (N - i) * 17 + (i % 31);
        }
    }

    std::vector<int> counts(static_cast<std::size_t>(nprocs));
    std::vector<int> displs(static_cast<std::size_t>(nprocs));
    for (int p = 0; p < nprocs; ++p) {
        counts[static_cast<std::size_t>(p)] = N / nprocs + (p < N % nprocs ? 1 : 0);
    }
    displs[0] = 0;
    for (int p = 1; p < nprocs; ++p) {
        displs[static_cast<std::size_t>(p)] =
            displs[static_cast<std::size_t>(p - 1)] +
            counts[static_cast<std::size_t>(p - 1)];
    }

    const int my_count = counts[static_cast<std::size_t>(rank)];
    std::vector<int> local(static_cast<std::size_t>(my_count));

    MPI_Scatterv(rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_INT,
                 local.data(), my_count, MPI_INT, 0, MPI_COMM_WORLD);

    merge_sort(local);

    int step = 1;
    while (step < nprocs) {
        const int tag_size = 7000 + step;
        const int tag_data = 8000 + step;

        if (rank % (2 * step) == 0) {
            const int partner = rank + step;
            if (partner < nprocs) {
                int recv_n = 0;
                MPI_Recv(&recv_n, 1, MPI_INT, partner, tag_size, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                std::vector<int> other(static_cast<std::size_t>(recv_n));
                if (recv_n > 0) {
                    MPI_Recv(other.data(), recv_n, MPI_INT, partner, tag_data, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                }
                std::vector<int> merged;
                merge_sorted_into(local, other, merged);
                local.swap(merged);
            }
        } else if (rank % (2 * step) == step) {
            const int dest = rank - step;
            int send_n = static_cast<int>(local.size());
            MPI_Send(&send_n, 1, MPI_INT, dest, tag_size, MPI_COMM_WORLD);
            if (send_n > 0) {
                MPI_Send(local.data(), send_n, MPI_INT, dest, tag_data, MPI_COMM_WORLD);
            }
            local.clear();
        }

        step *= 2;
    }

    if (rank == 0) {
        if (static_cast<int>(local.size()) != N) {
            std::cerr << "rank 0: expected " << N << " elements, got " << local.size() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        const bool ok = is_sorted(local);
        std::cout << "merge sort (MPI): N=" << N << " P=" << nprocs
                  << " — globally sorted: " << (ok ? "yes" : "NO") << "\n";
        if (N <= 32) {
            std::cout << "result: ";
            for (int i = 0; i < N; ++i) {
                std::cout << local[static_cast<std::size_t>(i)]
                          << (i + 1 < N ? ' ' : '\n');
            }
        }
    }

    MPI_Finalize();
    return 0;
}
