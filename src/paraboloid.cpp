#include "paraboloid.h"
#include "utils.h"

namespace jtrace {

    Paraboloid::Paraboloid(double _A, double _B) : A(_A), B(_B) {}

    std::string Paraboloid::repr() const {
        std::ostringstream oss(" ");
        oss << "Paraboloid(" << A << ", " << B << ")";
        return oss.str();
    }

    double Paraboloid::operator()(double x, double y) const {
        double r2 = x*x + y*y;
        return A*r2 + B;
    }

    Vec3 Paraboloid::normal(double x, double y) const {
        return Vec3(-2*A*x, -2*A*y, 1).UnitVec3();
    }

    Intersection Paraboloid::intersect(const Ray& ray) const {
        double a = A*(ray.v.x*ray.v.x + ray.v.y*ray.v.y);
        double b = 2*A*(ray.p0.x*ray.v.x + ray.p0.y*ray.v.y) - ray.v.z;
        double c = A*(ray.p0.x*ray.p0.x + ray.p0.y*ray.p0.y) + B - ray.p0.z;
        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

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
                }
            } else {
                if (r2 < 0) {
                    t = r1;
                } else {
                    t = std::min(r1, r2);
                }
            }
        }

        Vec3 point = ray(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal, this);
    }

    inline std::ostream& operator<<(std::ostream& os, const Paraboloid &p) {
        return os << p.repr();
    }

}
