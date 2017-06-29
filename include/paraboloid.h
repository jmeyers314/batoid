#ifndef jtrace_paraboloid_h
#define jtrace_paraboloid_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace jtrace {

    class Paraboloid : public Surface {
    public:
        Paraboloid(double _A, double _B,
                   double _Rin=0.0, double _Rout=std::numeric_limits<double>::infinity());
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        double getA() const {return A;}
        double getB() const {return B;}

        std::string repr() const;

    private:
        const double A, B;
        const double Rin, Rout;
    };

    inline std::ostream& operator<<(std::ostream& os, const Paraboloid& p);

}
#endif
