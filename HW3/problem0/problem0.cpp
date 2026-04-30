#include <cstdlib>
#include <iostream>
#include <string>

static int linear_count(int M, int P, int p) {
    const int base = M / P;
    const int rem = M % P;
    return base + (p < rem ? 1 : 0);
}

static int linear_offset(int M, int P, int p) {
    const int base = M / P;
    const int rem = M % P;
    // Sum counts for ranks [0, p)
    return p * base + (p < rem ? p : rem);
}

static bool linear_local_to_global(int M, int P, int p, int i, int& I) {
    if (M < 0 || P <= 0 || p < 0 || p >= P || i < 0) {
        return false;
    }
    const int cnt = linear_count(M, P, p);
    if (i >= cnt) {
        return false;
    }
    I = linear_offset(M, P, p) + i;
    return true;
}

static bool scatter_global_to_local(int M, int P, int I, int& p_out, int& i_out) {
    if (M < 0 || P <= 0 || I < 0 || I >= M) {
        return false;
    }
    p_out = I % P;
    i_out = I / P;
    return true;
}

int main(int argc, char** argv) {
    // Problem statement asks for M, P, p, i as input.
    // Usage: ./problem0 M P p i
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " M P p i\n";
        std::cerr << "Example: " << argv[0] << " 8 4 0 1\n";
        return 1;
    }

    const int M = std::atoi(argv[1]);
    const int P = std::atoi(argv[2]);
    const int p = std::atoi(argv[3]);
    const int i = std::atoi(argv[4]);

    if (M < 0 || P <= 0) {
        std::cerr << "Error: require M >= 0 and P > 0.\n";
        return 1;
    }
    if (p < 0 || p >= P) {
        std::cerr << "Error: process rank p must satisfy 0 <= p < P.\n";
        return 1;
    }

    const int local_count_linear = linear_count(M, P, p);
    int I = -1;
    if (!linear_local_to_global(M, P, p, i, I)) {
        std::cerr << "Error: local index i is invalid for linear distribution on rank p.\n";
        std::cerr << "For rank p=" << p << ", valid i range is [0, "
                  << (local_count_linear > 0 ? local_count_linear - 1 : -1) << "].\n";
        return 1;
    }

    int p_scatter = -1;
    int i_scatter = -1;
    if (!scatter_global_to_local(M, P, I, p_scatter, i_scatter)) {
        std::cerr << "Error: failed to map global index to scatter distribution.\n";
        return 1;
    }

    std::cout << "Inputs: M=" << M << ", P=" << P << ", p=" << p << ", i=" << i << "\n";
    std::cout << "Linear distribution local count on rank p: " << local_count_linear << "\n";
    std::cout << "Global index I from linear inverse mapping: " << I << "\n";
    std::cout << "Scatter mapping result: p'=" << p_scatter << ", i'=" << i_scatter << "\n";

    // Optional quick sanity hint for the figure example from the handout.
    if (M == 8 && P == 4 && p == 0 && i == 1) {
        std::cout << "(Figure-like check) Expected I=1, scatter => p'=1, i'=0\n";
    }

    return 0;
}
