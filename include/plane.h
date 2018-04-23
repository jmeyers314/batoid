#ifndef batoid_plane_h
#define batoid_plane_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Plane : public Surface {
    public:
        virtual double sag(double, double) const { return 0.0; }
        virtual Vector3d normal(double, double) const { return Vector3d(0.0, 0.0, 1.0); }
        virtual Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        std::string repr() const;
    };

    inline bool operator==(const Plane& p1, const Plane& p2) { return true; }
    inline bool operator!=(const Plane& p1, const Plane& p2) { return false; }
}
#endif
