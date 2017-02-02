#include "plane.h"

namespace jtrace {
    Plane::Plane(double _B) : B(_B) {}

    Intersection Plane::intersect(const Ray &r) const {
        double t = (B - r.p0.z)/r.v.z;
        Vec3 point = r(t);
        Vec3 surfaceNormal = normal(point.x, point.y);
        return Intersection(t, point, surfaceNormal, this);
    }

    std::string Plane::repr() const {
        std::ostringstream oss (" ");
        oss << "Plane(" << B << ")";
        return oss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const Plane &p) {
        return os << p.repr();
    }
}
