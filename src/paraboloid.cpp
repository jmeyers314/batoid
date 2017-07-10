#include "paraboloid.h"
#include "utils.h"
#include "except.h"
#include <cmath>

namespace jtrace {

    Paraboloid::Paraboloid(double _R, double _B, double _Rin, double _Rout) :
        R(_R), B(_B), Rin(_Rin), Rout(_Rout) {}

    std::string Paraboloid::repr() const {
        std::ostringstream oss(" ");
        oss << "Paraboloid(" << R << ", " << B << ")";
        return oss.str();
    }

    double Paraboloid::sag(double x, double y) const {
        double r2 = x*x + y*y;
        double result = B;
        if (R != 0)
            result += r2/(2*R);
        return result;
    }

    Vec3 Paraboloid::normal(double x, double y) const {
        return Vec3(-x/R, -y/R, 1).UnitVec3();
    }

    Intersection Paraboloid::intersect(const Ray& ray) const {
        if (ray.failed)
            return Intersection(true);
        double a = (ray.v.x*ray.v.x + ray.v.y*ray.v.y)/2/R;
        double b = (ray.p0.x*ray.v.x + ray.p0.y*ray.v.y)/R - ray.v.z;
        double c = (ray.p0.x*ray.p0.x + ray.p0.y*ray.p0.y)/2/R + B - ray.p0.z;
        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

        double t;
        if (n == 0) {
            // throw NoIntersectionError("");
            return Intersection(true);
        } else if (n == 1) {
            if (r1 < 0) {
                // throw NoFutureIntersectionError("");
                return Intersection(true);
            }
            t = r1;
        } else {
            if (r1 < 0) {
                if (r2 < 0) {
                    // throw NoFutureIntersectionError("");
                    return Intersection(true);
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

        t += ray.t0;
        Vec3 point = ray.positionAtTime(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        double rho = std::hypot(point.x, point.y);
        bool isVignetted = rho < Rin || rho > Rout;
        return Intersection(t, point, surfaceNormal, isVignetted);
    }

    inline std::ostream& operator<<(std::ostream& os, const Paraboloid& p) {
        return os << p.repr();
    }

}
