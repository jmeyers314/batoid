#ifndef __jem_intersection__h
#define __jem_intersection__h

#include <string>
#include <sstream>
#include "vec3.h"
#include "surface.h"

namespace jtrace {
    class Surface;
    struct Intersection {
        Intersection(double _t, Vec3 _point, Vec3 _surfaceNormal, const Surface* _surface);

        double t;
        Vec3 point;
        Vec3 surfaceNormal;
        const Surface* surface; // Pointer to surface that was intersected.

        std::string repr() const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Intersection &i) {
        return os << i.repr();
    }
}

#endif
