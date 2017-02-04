#ifndef __jem_intersection__h
#define __jem_intersection__h

#include <string>
#include <sstream>
#include "vec3.h"
#include "surface.h"

namespace jtrace {
    class Surface;
    struct Intersection {
        Intersection(const double _t, const Vec3 _point,
                     const Vec3 _surfaceNormal, const Surface* _surface);

        const double t;
        const Vec3 point;
        const Vec3 surfaceNormal;
        const Surface* surface; // Pointer to surface that was intersected.

        std::string repr() const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Intersection &i) {
        return os << i.repr();
    }
}

#endif
