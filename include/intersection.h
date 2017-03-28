#ifndef jtrace_intersection_h
#define jtrace_intersection_h

#include <string>
#include <sstream>
#include "vec3.h"
#include "ray.h"

namespace jtrace {
    struct Intersection {
        Intersection(const double _t, const Vec3 _point, const Vec3 _surfaceNormal);

        double t;
        Vec3 point;
        Vec3 surfaceNormal;

        double getX0() const { return point.x; }
        double getY0() const { return point.y; }
        double getZ0() const { return point.z; }

        double getNx() const { return surfaceNormal.x; }
        double getNy() const { return surfaceNormal.y; }
        double getNz() const { return surfaceNormal.z; }

        Ray reflectedRay(const Ray&) const;
        Ray refractedRay(const Ray&, double n1, double n2) const;
        std::string repr() const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Intersection& i) {
        return os << i.repr();
    }

    inline bool operator==(const Intersection& i1, const Intersection& i2) {
        return i1.t == i2.t && i1.point == i2.point && i1.surfaceNormal == i2.surfaceNormal;
    }
}

#endif
