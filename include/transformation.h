#ifndef __jem_transformation__h
#define __jem_transformation__h

#include "jtrace.h"

namespace jtrace {
    struct Intersection;
    class Surface;
    class Transformation : public Surface {
    public:
        Transformation(const Surface*, double dx, double dy, double dz);
        virtual double operator()(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Intersection intersect(const Ray&) const;
        std::string repr() const;
        double getDx() const { return dr.x; }
        double getDy() const { return dr.y; }
        double getDz() const { return dr.z; }
    private:
        const Surface* transformee;
        const Vec3 dr;
    };

    inline std::ostream& operator<<(std::ostream& os, const Transformation& t);
}

#endif
