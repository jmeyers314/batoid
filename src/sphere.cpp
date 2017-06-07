#include "sphere.h"
#include "utils.h"
#include "except.h"

namespace jtrace {
    Sphere::Sphere(double _R, double _B, double _Rin, double _Rout) :
        R(_R), B(_B), Rin(_Rin), Rout(_Rout) {}

    double Sphere::operator()(double x, double y) const {
        double r2 = x*x + y*y;
        double result = B;
        if (R != 0) {
            result += R*(1-std::sqrt(1-r2/R/R));
        }
        return result;
    }

    Vec3 Sphere::normal(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        double dzdr1 = dzdr(r);
        Vec3 n{-dzdr1*x/r, -dzdr1*y/r, 1};
        return n.UnitVec3();
    }

    Intersection Sphere::intersect(const Ray &r) const {
        double vr2 = r.v.x*r.v.x + r.v.y*r.v.y;
        double vz2 = r.v.z*r.v.z;
        double vrr0 = r.v.x*r.p0.x + r.v.y*r.p0.y;
        double r02 = r.p0.x*r.p0.x + r.p0.y*r.p0.y;
        double z0term = (r.p0.z-B-R);

        // Quadratic equation coefficients
        double a = vz2 + vr2;
        double b = 2*r.v.z*z0term + 2*vrr0;
        double c = z0term*z0term - R*R + r02;

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
        double rho = std::hypot(point.x, point.y);
        bool isVignetted = rho < Rin || rho > Rout;
        return Intersection(t, point, surfaceNormal, isVignetted);
    }

    std::string Sphere::repr() const {
        std::ostringstream oss(" ");
        oss << "Sphere(" << R << ", " << B << ")";
        return oss.str();
    }

    double Sphere::dzdr(double r) const {
        double rat = r/R;
        return rat/std::sqrt(1-rat*rat);
    }

    inline std::ostream& operator<<(std::ostream& os, const Sphere& q) {
        return os << q.repr();
    }

}
