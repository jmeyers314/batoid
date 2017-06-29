#include "plane.h"
#include <cmath>

namespace jtrace {
    Plane::Plane(double _B, double _Rin, double _Rout) :
        B(_B), Rin(_Rin), Rout(_Rout) {}

    Intersection Plane::intersect(const Ray& r) const {
        if (r.failed)
            return Intersection(true);
        double t = (B - r.p0.z)/r.v.z;
        t += r.t0;
        Vec3 point = r.positionAtTime(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        double rho = std::hypot(point.x, point.y);
        bool isVignetted = rho < Rin || rho > Rout;
        return Intersection(t, point, surfaceNormal, isVignetted);
    }

    std::string Plane::repr() const {
        std::ostringstream oss (" ");
        oss << "Plane(" << B << ")";
        return oss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const Plane& p) {
        return os << p.repr();
    }
}
