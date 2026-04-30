// Attempt at HW3 graduate Problem G2 (single-rank Kokkos kernel version).
// Uses halos and ping-pong domains to avoid boundary if-statements in the kernel.

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const int rows = (argc > 1) ? std::atoi(argv[1]) : 16;
        const int cols = (argc > 2) ? std::atoi(argv[2]) : 16;
        const int steps = (argc > 3) ? std::atoi(argv[3]) : 20;
        if (rows <= 0 || cols <= 0 || steps < 0) {
            std::cerr << "Usage: " << argv[0] << " [rows cols steps], rows/cols > 0\n";
            Kokkos::finalize();
            return 1;
        }

        // Include 1-cell halo on each side.
        Kokkos::View<int**> even("even", rows + 2, cols + 2);
        Kokkos::View<int**> odd("odd", rows + 2, cols + 2);
        Kokkos::deep_copy(even, 0);
        Kokkos::deep_copy(odd, 0);

        // Seed a glider in the interior.
        auto h_even = Kokkos::create_mirror_view(even);
        h_even(2, 3) = 1;
        h_even(3, 4) = 1;
        h_even(4, 2) = 1;
        h_even(4, 3) = 1;
        h_even(4, 4) = 1;
        Kokkos::deep_copy(even, h_even);

        for (int t = 0; t < steps; ++t) {
            auto src = (t % 2 == 0) ? even : odd;
            auto dst = (t % 2 == 0) ? odd : even;

            // Update periodic halos on host for readability.
            auto h_src = Kokkos::create_mirror_view(src);
            Kokkos::deep_copy(h_src, src);
            for (int j = 1; j <= cols; ++j) {
                h_src(0, j) = h_src(rows, j);       // top halo
                h_src(rows + 1, j) = h_src(1, j);   // bottom halo
            }
            for (int i = 1; i <= rows; ++i) {
                h_src(i, 0) = h_src(i, cols);       // left halo
                h_src(i, cols + 1) = h_src(i, 1);   // right halo
            }
            h_src(0, 0) = h_src(rows, cols);
            h_src(0, cols + 1) = h_src(rows, 1);
            h_src(rows + 1, 0) = h_src(1, cols);
            h_src(rows + 1, cols + 1) = h_src(1, 1);
            Kokkos::deep_copy(src, h_src);

            // Interior-only update: no boundary if-statements required.
            Kokkos::parallel_for(
                "gol_step", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {rows + 1, cols + 1}),
                KOKKOS_LAMBDA(int i, int j) {
                    const int neighbors = src(i - 1, j - 1) + src(i - 1, j) + src(i - 1, j + 1) +
                                          src(i, j - 1) + src(i, j + 1) +
                                          src(i + 1, j - 1) + src(i + 1, j) + src(i + 1, j + 1);
                    const int alive = src(i, j);
                    dst(i, j) = (alive == 1) ? ((neighbors == 2 || neighbors == 3) ? 1 : 0)
                                             : (neighbors == 3 ? 1 : 0);
                });
            Kokkos::fence();
        }

        auto final_grid = (steps % 2 == 0) ? even : odd;
        auto h_final = Kokkos::create_mirror_view(final_grid);
        Kokkos::deep_copy(h_final, final_grid);

        int live_count = 0;
        for (int i = 1; i <= rows; ++i) {
            for (int j = 1; j <= cols; ++j) live_count += h_final(i, j);
        }

        std::cout << "Game of Life (Kokkos) complete: rows=" << rows
                  << " cols=" << cols << " steps=" << steps << "\n";
        std::cout << "Live-cell count after final step: " << live_count << "\n";
        std::cout << "Note: this is a single-process Kokkos-kernel attempt with halos and even/odd grids.\n";
    }
    Kokkos::finalize();
    return 0;
}
