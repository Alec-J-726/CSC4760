#include <iomanip>
#include <iostream>

int main() {
    constexpr int n = 100000;
    constexpr double tiny = 1e-18;

    // Case 1: start at 1.0, repeatedly add tiny increment.
    double sum_from_one = 1.0;
    for (int k = 0; k < n; ++k) {
        sum_from_one += tiny;
    }

    // Case 2: accumulate tiny values first, then add 1.0.
    double sum_from_zero = 0.0;
    for (int k = 0; k < n; ++k) {
        sum_from_zero += tiny;
    }
    const double one_plus_accum = 1.0 + sum_from_zero;

    const double diff = one_plus_accum - sum_from_one;

    std::cout << std::setprecision(20);
    std::cout << "n = " << n << ", tiny = " << tiny << "\n\n";

    std::cout << "sum_from_one       = " << sum_from_one << "\n";
    std::cout << "sum_from_zero      = " << sum_from_zero << "\n";
    std::cout << "one_plus_accum     = " << one_plus_accum << "\n";
    std::cout << "difference         = " << diff << "\n\n";

    std::cout << std::hexfloat;
    std::cout << "hex(sum_from_one)  = " << sum_from_one << "\n";
    std::cout << "hex(one_plus_accum)= " << one_plus_accum << "\n";
    std::cout << "hex(difference)    = " << diff << "\n";
    std::cout << std::defaultfloat;

    return 0;
}
