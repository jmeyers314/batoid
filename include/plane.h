#ifndef batoid_plane_h
#define batoid_plane_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {

    class Plane : public Surface {
    public:
        virtual double sag(double, double) const { return 0.0; }
        virtual Vec3 normal(double, double) const { return Vec3(0,0,1); }
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        virtual Ray intercept(const Ray&) const;
        virtual void interceptInPlace(Ray&) const;

        std::string repr() const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Plane& p);
}
#endif
