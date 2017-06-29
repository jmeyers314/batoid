#ifndef jtrace_sphere_h
#define jtrace_sphere_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace jtrace {

    class Sphere : public Surface {
    public:
        Sphere(double _R, double _B,
                double _Rin=0.0, double _Rout=std::numeric_limits<double>::infinity());
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        double getR() const {return R;}
        double getB() const {return B;}

        std::string repr() const;

    private:
        const double R, B;
        const double Rin, Rout;

        double dzdr(double r) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Sphere& q);

}
#endif
