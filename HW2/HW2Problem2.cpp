#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Initialize a View; the first constructor argument is the label used by .label().
        Kokkos::View<double*> data("HW2Problem2_data", 16);

        std::cout << "View label: " << data.label() << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
