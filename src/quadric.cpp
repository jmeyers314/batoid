#include <iostream>
#include "quadric.h"

namespace jtrace {
    Quadric::Quadric(double _R, double _kappa, double _B) :
        R(_R), kappa(_kappa), B(_B) {}

    double Quadric::operator()(double x, double y) const {
        double r2 = x*x + y*y;
        double den = R*(1.+std::sqrt(1.-(1.+kappa)*r2/R/R));
        return r2/den + B;

        // Following also works, except leads to divide by 0 when kappa=-1
        // return R/(1+kappa)*(1-std::sqrt(1-(1+kappa)*r2/R/R))+B;
    }

    Vec3 Quadric::normal(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        double dzdr1 = dzdr(r);
        Vec3 n{-dzdr1*x/r, -dzdr1*y/r, 1};
        return n.UnitVec3();
    }

    Intersection Quadric::intersect(const Ray &r) const {
        double vr2 = r.v.x*r.v.x + r.v.y*r.v.y;
        double vz2 = r.v.z*r.v.z;
        double vrr0 = r.v.x*r.p0.x + r.v.y*r.p0.y;
        double r02 = r.p0.x*r.p0.x + r.p0.y*r.p0.y;
        double z0term = (r.p0.z-B-R/(1+kappa));

        // Quadratic equation coefficients
        double a = vz2 + vr2/(1+kappa);
        double b = 2*r.v.z*z0term + 2*vrr0/(1+kappa);
        double c = z0term*z0term - R*R/(1+kappa)/(1+kappa) + r02/(1+kappa);

        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

        // Should probably check the solutions here since we obtained the quadratic
        // formula above by squaring both sides of an equation.

        double t;
        if (n == 0) {
            throw NoIntersectionError("");
        } else if (n == 1) {
            if (r1 < 0) {
                throw NoFutureIntersectionError("");
            }
            t = r1;
        } else {
            if (r1 < 0) {
                if (r2 < 0) {
                    throw NoFutureIntersectionError("");
                } else {
                    t = r2;
                }
            } else {
                if (r2 < 0) {
                    t = r1;
                } else {
                    t = std::min(r1, r2);
                }
            }
        }

        t += r.t0;
        Vec3 point = r(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal);
    }

    std::string Quadric::repr() const {
        std::ostringstream oss(" ");
        oss << "Quadric(" << R << ", " << kappa << ", " << B << ")";
        return oss.str();
    }

    double Quadric::dzdr(double r) const {
        return r/(R*std::sqrt(1-r*r*(1+kappa)/R/R));
    }

    inline std::ostream& operator<<(std::ostream& os, const Quadric &a) {
        return os << a.repr();
    }

}
