// I did not test this; it is only an attempt at extra credit (grad Problem 9).
// (CSC 5760 — not required for undergrad CSC 4760.)

#include <Kokkos_Core.hpp>

#include <iostream>

struct OnesInclusivePrefixScan {
    Kokkos::View<int*> a;
    using value_type = int;
    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const { update = 0; }
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, int& update, const bool final) const {
        const int val = a(i);
        if (final) {
            a(i) = update + val;
        }
        update += val;
    }
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int n = 16;
        Kokkos::View<int*> a("a", n);
        Kokkos::deep_copy(a, 1);

        constexpr int num_runs = 3;
        for (int r = 0; r < num_runs; ++r) {
            Kokkos::deep_copy(a, 1);
            Kokkos::Timer timer;
            Kokkos::parallel_scan(
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
                OnesInclusivePrefixScan{a});
            Kokkos::fence();
            const double sec = timer.seconds();

            auto h = Kokkos::create_mirror_view(a);
            Kokkos::deep_copy(h, a);

            std::cout << "run " << r << "  parallel_scan time (s): " << sec
                      << "\npartial sums: ";
            for (int i = 0; i < n; ++i) {
                std::cout << h(i) << (i + 1 < n ? ' ' : '\n');
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
