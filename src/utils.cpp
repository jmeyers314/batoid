#include "utils.h"
#include <cmath>
#include <utility>

namespace batoid {

    constexpr double TOLERANCE = 1.0e-15;

    inline bool IsZero(double x) { return (fabs(x) < TOLERANCE); }

    int solveQuadratic(double a, double b, double c, double& r1, double& r2) {
        if (IsZero(a)) {
            // No quadratic term.  Linear equation.
            if (IsZero(b)) {
                // Solve c=0 for x.  Ack!
                return 0;
            } else {
                r1 = -c / b;
                return 1;
            }
        } else if (IsZero(c)) {
            // No constant term.
            r1 = 0;
            r2 = -b / a;
            if (r2 < r1) {
                std::swap(r1, r2);
            }
            return 2;
        } else {
            const double discriminant = b*b - 4*a*c;
            if (IsZero(discriminant)) {
                // Degenerate case.  Should I return 1 or 2?
                r1 = r2 = -b / (2*a);
                return 1;
            } else if (discriminant < 0) {
                // Results would be complex.  This version only works for real results.
                return 0;
            } else {
                // Avoid subtracting similar valued numbers (abs(b) and sqrt(discriminant)).
                if (b > 0) {
                    r1 = (-b - std::sqrt(discriminant)) / (2*a);
                } else {
                    r1 = 2*c / (-b + std::sqrt(discriminant));
                }
                r2 = c / (a*r1);
                if (r2 < r1) {
                    std::swap(r1, r2);
                }
                return 2;
            }
        }
    }
}
