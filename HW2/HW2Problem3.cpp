#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int n = 4;
        constexpr int m = 5;

        // n × m View; element at (i, j) is 1000 * i * j (0-based indices).
        Kokkos::View<int**> A("HW2Problem3", n, m);

        Kokkos::parallel_for(
            "fill_HW2Problem3",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
            KOKKOS_LAMBDA(int i, int j) { A(i, j) = 1000 * i * j; });

        Kokkos::fence();

        // Host mirror so printing works whether the default space is host or device.
        auto h_A = Kokkos::create_mirror_view(A);
        Kokkos::deep_copy(h_A, A);

        std::cout << "n = " << n << ", m = " << m << "\n";
        std::cout << "View label: " << A.label() << "\n";
        std::cout << "Contents (each entry = 1000 * i * j):\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                std::cout << h_A(i, j);
                if (j + 1 < m) {
                    std::cout << '\t';
                }
            }
            std::cout << '\n';
        }
    }
    Kokkos::finalize();
    return 0;
}
