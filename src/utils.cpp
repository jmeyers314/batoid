#include "utils.h"
#include <cmath>

namespace jtrace {

    constexpr double TOLERANCE = 1.0e-8;

    inline bool IsZero(double x) { return (fabs(x) < TOLERANCE); }

    int solveQuadratic(double a, double b, double c, double& r1, double& r2) {
        if (IsZero(a)) {
            if (IsZero(b)) {
                return 0;
            } else {
                r1 = -c / b;
                return 1;
            }
        } else {
            const double discriminant = b*b - 4.0*a*c;
            if (IsZero(discriminant)) {
                r1 = r2 = -b / (2.0 * a);
                return 1;
            } else if (discriminant < 0.0) {
                return 0;
            } else {
                const double r = std::sqrt(discriminant);
                const double d = 2*a;
                r1 = (-b + r) / d;
                r2 = (-b - r) / d;
                return 2;
            }
        }
    }

}
