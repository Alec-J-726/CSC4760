#include <Kokkos_Core.hpp>
#include <iostream>

// Extra credit: verify shapes for C(i,j) = A(i,j) + B(j).
void check_add_shapes(const Kokkos::View<const double**>& A,
                      const Kokkos::View<const double*>& B,
                      const Kokkos::View<double**>& C) {
    if (A.extent(0) != C.extent(0) || A.extent(1) != C.extent(1)) {
        Kokkos::abort("HW2Problem7: A and C must have the same shape.");
    }
    if (A.extent(1) != B.extent(0)) {
        Kokkos::abort("HW2Problem7: number of columns of A must equal length of B.");
    }
}

// Parallel update: C(i,j) = A(i,j) + B(j) (add vector B along each row).
void add_vector_to_each_row(const Kokkos::View<const double**>& A,
                            const Kokkos::View<const double*>& B,
                            const Kokkos::View<double**>& C) {
    check_add_shapes(A, B, C);
    const int nrows = static_cast<int>(A.extent(0));
    const int ncols = static_cast<int>(A.extent(1));
    Kokkos::parallel_for(
        "HW2Problem7_add_vector_to_rows",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
        KOKKOS_LAMBDA(int i, int j) { C(i, j) = A(i, j) + B(j); });
    Kokkos::fence();
}

static void print_matrix_host(const Kokkos::View<double**, Kokkos::HostSpace>& M,
                              const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < static_cast<int>(M.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(M.extent(1)); ++j) {
            std::cout << M(i, j);
            if (j + 1 < static_cast<int>(M.extent(1))) {
                std::cout << '\t';
            }
        }
        std::cout << '\n';
    }
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int nrows = 3;
        constexpr int ncols = 3;

        Kokkos::View<double**> A("A", nrows, ncols);
        Kokkos::View<double*> B("B", ncols);
        Kokkos::View<double**> C("C", nrows, ncols);

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

        add_vector_to_each_row(A, B, C);

        auto h_C = Kokkos::create_mirror_view(C);
        Kokkos::deep_copy(h_C, C);

        std::cout << "Test case from homework (A, B, then C = A + B(j) per row):\n";
        print_matrix_host(h_A, "A");
        std::cout << "B (row vector):\t" << h_B(0) << '\t' << h_B(1) << '\t' << h_B(2) << "\n\n";
        print_matrix_host(h_C, "C");

        // Expected solution from assignment
        Kokkos::View<double**>::HostMirror expected("expected", nrows, ncols);
        expected(0, 0) = 351;
        expected(0, 1) = 159;
        expected(0, 2) = 272;
        expected(1, 0) = 445;
        expected(1, 1) = 170;
        expected(1, 2) = 344;
        expected(2, 0) = 275;
        expected(2, 1) = 170;
        expected(2, 2) = 277;

        double err = 0.0;
        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < ncols; ++j) {
                err = Kokkos::max(err, Kokkos::abs(h_C(i, j) - expected(i, j)));
            }
        }
        std::cout << "Max abs error vs expected Soln: " << err << "\n";
    }
    Kokkos::finalize();
    return 0;
}
