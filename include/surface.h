#ifndef __jem_surface__h
#define __jem_surface__h

#include "vec3.h"
#include "ray.h"
#include "intersection.h"

namespace jtrace {
    struct Intersection;
    class Transformation;
    class Surface : public std::enable_shared_from_this<Surface> {
    public:
        virtual double operator()(double, double) const = 0;
        virtual Vec3 normal(double, double) const = 0;
        virtual Intersection intersect(const Ray&) const = 0;
        virtual std::string repr() const = 0;
        Transformation shift(double dx, double dy, double dz) const;
        Transformation rotX(double theta) const;
        Transformation rotY(double theta) const;
        Transformation rotZ(double theta) const;
    };

}
#endif
