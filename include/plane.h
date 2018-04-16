#ifndef batoid_plane_h
#define batoid_plane_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {

    class Plane : public Surface {
    public:
        virtual double sag(double, double) const { return 0.0; }
        virtual Vec3 normal(double, double) const { return Vec3(0,0,1); }
        virtual Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        std::string repr() const;
    };

    inline bool operator==(const Plane& p1, const Plane& p2) { return true; }
    inline bool operator!=(const Plane& p1, const Plane& p2) { return false; }
}
#endif
