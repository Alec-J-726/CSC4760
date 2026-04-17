#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int nrows = 4096;
        constexpr int ncols = 4096;

        Kokkos::View<double**> A("A", nrows, ncols);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
            KOKKOS_LAMBDA(int i, int j) { A(i, j) = static_cast<double>(i + j); });
        Kokkos::fence();

        Kokkos::View<double*> row_sum_parallel("row_sum_parallel", nrows);
        Kokkos::View<double*> row_sum_serial("row_sum_serial", nrows);

        Kokkos::Timer timer;

        timer.reset();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, nrows),
            KOKKOS_LAMBDA(int i) {
                double s = 0.0;
                for (int j = 0; j < ncols; ++j) {
                    s += A(i, j);
                }
                row_sum_parallel(i) = s;
            });
        Kokkos::fence();
        const double t_parallel = timer.seconds();

        auto h_A = Kokkos::create_mirror_view(A);
        Kokkos::deep_copy(h_A, A);

        timer.reset();
        for (int i = 0; i < nrows; ++i) {
            double s = 0.0;
            for (int j = 0; j < ncols; ++j) {
                s += h_A(i, j);
            }
            row_sum_serial(i) = s;
        }
        const double t_serial = timer.seconds();

        // Optional sanity check (first row)
        auto h_par = Kokkos::create_mirror_view(row_sum_parallel);
        auto h_ser = Kokkos::create_mirror_view(row_sum_serial);
        Kokkos::deep_copy(h_par, row_sum_parallel);
        Kokkos::deep_copy(h_ser, row_sum_serial);

        std::cout << "Matrix: " << nrows << " x " << ncols << "\n";
        std::cout << "Kokkos parallel_for (row sums): " << t_parallel << " s\n";
        std::cout << "Standard nested for (row sums): " << t_serial << " s\n";
        std::cout << "Check row 0 | parallel: " << h_par(0) << "  serial: " << h_ser(0)
                  << "\n";
    }
    Kokkos::finalize();
    return 0;
}
