#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Problem 4: a 5 × 7 × 12 × n View (last dimension is given the name n in the assignment).
        constexpr int n = 4;

        Kokkos::View<int****> A("HW2Problem4", 5, 7, 12, n);

        std::cout << "View label: " << A.label() << "\n";
        std::cout << "Extents: ";
        for (int r = 0; r < static_cast<int>(A.rank()); ++r) {
            std::cout << A.extent(r);
            if (r + 1 < static_cast<int>(A.rank())) {
                std::cout << " × ";
            }
        }
        std::cout << "  (5 × 7 × 12 × " << n << ")\n";
        std::cout << "Total elements: " << A.size() << "\n";
    }
    Kokkos::finalize();
    return 0;
}
