#ifndef batoid_surface_h
#define batoid_surface_h

#include <vector>
#include <memory>
#include "vec3.h"
#include "ray.h"
#include "intersection.h"

namespace batoid {
    struct Intersection;
    class Transformation;
    class Surface : public std::enable_shared_from_this<Surface> {
    public:
        virtual double sag(double, double) const = 0;
        virtual Vec3 normal(double, double) const = 0;

        virtual Intersection intersect(const Ray&) const = 0;
        virtual Ray intercept(const Ray&) const = 0;
        virtual void interceptInPlace(Ray&) const = 0;

        std::vector<Intersection> intersect(const std::vector<Ray>&) const;
        std::vector<Ray> intercept(const std::vector<Ray>&) const;
        void interceptInPlace(std::vector<Ray>&) const;

        virtual std::string repr() const = 0;
        Transformation shift(double dx, double dy, double dz) const;
        Transformation shift(const Vec3&) const;
        Transformation rotX(double theta) const;
        Transformation rotY(double theta) const;
        Transformation rotZ(double theta) const;
    };
}
#endif
