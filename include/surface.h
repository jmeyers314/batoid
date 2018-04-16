#ifndef batoid_surface_h
#define batoid_surface_h

#include <vector>
#include <memory>
#include "vec3.h"
#include "ray.h"

namespace batoid {
    class Surface : public std::enable_shared_from_this<Surface> {
    public:
        virtual double sag(double, double) const = 0;
        virtual Vec3 normal(double, double) const = 0;

        virtual Ray intersect(const Ray&) const = 0;
        virtual void intersectInPlace(Ray&) const = 0;

        std::vector<Ray> intersect(const std::vector<Ray>&) const;
        void intersectInPlace(std::vector<Ray>&) const;

        virtual std::string repr() const = 0;
    };
    std::ostream& operator<<(std::ostream& os, const Surface& s);
}
#endif
