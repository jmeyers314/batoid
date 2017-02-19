#ifndef __jem_intersection__h
#define __jem_intersection__h

#include <string>
#include <sstream>
#include "vec3.h"
#include "surface.h"

namespace jtrace {
    class Surface;
    struct Intersection {
        Intersection(const double _t, const Vec3 _point, const Vec3 _surfaceNormal);

        const double t;
        const Vec3 point;
        const Vec3 surfaceNormal;

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
}

#endif
