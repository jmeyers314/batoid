#ifndef batoid_surface_h
#define batoid_surface_h

#include <vector>
#include <memory>
#include "ray.h"

#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    class Surface {
    public:
        virtual double sag(double, double) const = 0;
        virtual Vector3d normal(double, double) const = 0;

        virtual Ray intersect(const Ray&) const = 0;
        virtual void intersectInPlace(Ray&) const = 0;

        std::vector<Ray> intersect(const std::vector<Ray>&) const;
        void intersectInPlace(std::vector<Ray>&) const;

        virtual std::string repr() const = 0;
    };
    std::ostream& operator<<(std::ostream& os, const Surface& s);
}
#endif
