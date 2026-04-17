// I did not test this; it is only an attempt at extra credit (grad Problem 10).
// (CSC 5760 — not required for undergrad CSC 4760.)

#include <Kokkos_Core.hpp>

#include <iostream>

void check_matvec_shapes(const Kokkos::View<const double**>& A,
                         const Kokkos::View<const double*>& x,
                         const Kokkos::View<double*>& y) {
    if (A.extent(1) != x.extent(0)) {
        Kokkos::abort("HW2Problem10: columns of A must match length of x.");
    }
    if (A.extent(0) != y.extent(0)) {
        Kokkos::abort("HW2Problem10: rows of A must match length of y.");
    }
}

void matrix_vector_multiply(const Kokkos::View<const double**>& A,
                            const Kokkos::View<const double*>& x,
                            const Kokkos::View<double*>& y) {
    check_matvec_shapes(A, x, y);
    const int nrows = static_cast<int>(A.extent(0));
    const int ncols = static_cast<int>(A.extent(1));
    Kokkos::parallel_for(
        "HW2Problem10_matvec",
        nrows,
        KOKKOS_LAMBDA(int i) {
            double s = 0.0;
            for (int j = 0; j < ncols; ++j) {
                s += A(i, j) * x(j);
            }
            y(i) = s;
        });
    Kokkos::fence();
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int nrows = 3;
        constexpr int ncols = 3;

        Kokkos::View<double**> A("A", nrows, ncols);
        Kokkos::View<double*> B("B", ncols);
        Kokkos::View<double*> y("y", nrows);

        auto h_A = Kokkos::create_mirror_view(A);
        h_A(0, 0) = 130;
        h_A(0, 1) = 147;
        h_A(0, 2) = 115;
        h_A(1, 0) = 224;
        h_A(1, 1) = 158;
        h_A(1, 2) = 187;
        h_A(2, 0) = 54;
        h_A(2, 1) = 158;
        h_A(2, 2) = 120;

        auto h_B = Kokkos::create_mirror_view(B);
        h_B(0) = 221;
        h_B(1) = 12;
        h_B(2) = 157;

        Kokkos::deep_copy(A, h_A);
        Kokkos::deep_copy(B, h_B);

        matrix_vector_multiply(A, B, y);

        auto h_y = Kokkos::create_mirror_view(y);
        Kokkos::deep_copy(h_y, y);

        std::cout << "y = A * B (Problem 7 test views):\n";
        for (int i = 0; i < nrows; ++i) {
            std::cout << h_y(i) << (i + 1 < nrows ? '\t' : '\n');
        }
    }
    Kokkos::finalize();
    return 0;
}
