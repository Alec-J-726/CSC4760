#include <Kokkos_Core.hpp>
#include <iostream>
#include <limits>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int n = 32;

        Kokkos::View<double*> v("HW2Problem5", n);

        // Deterministic, non-random fill (assignment: do not use random elements).
        Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(int i) {
                v(i) = (i == n - 1) ? 1000.0 + static_cast<double>(i)
                                    : static_cast<double>(i);
            });
        Kokkos::fence();

        const double expected_max = 1000.0 + static_cast<double>(n - 1);

        double max_val = std::numeric_limits<double>::lowest();
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(int i, double& thread_max) {
                thread_max = Kokkos::max(thread_max, v(i));
            },
            Kokkos::Max<double>(max_val));
        Kokkos::fence();

        std::cout << "Maximum element (parallel_reduce): " << max_val << "\n";
        std::cout << "Expected max: " << expected_max << "\n";
    }
    Kokkos::finalize();
    return 0;
}
