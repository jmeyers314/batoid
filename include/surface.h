#ifndef __jem_surface__h
#define __jem_surface__h

#include "vec3.h"
#include "ray.h"
#include "intersection.h"

namespace jtrace {

    class Surface {
    public:
        virtual double operator()(double, double) const = 0;
        virtual Vec3 normal(double, double) const = 0;
        virtual Intersection intersect(const Ray&) const = 0;
    };

}
#endif
