#ifndef batoid_quadric_h
#define batoid_quadric_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {

    class Quadric : public Surface {
    public:
        Quadric(double _R, double _kappa, double _B,
                double _Rin=0.0, double _Rout=std::numeric_limits<double>::infinity());
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        double getR() const {return R;}
        double getKappa() const {return kappa;}
        double getB() const {return B;}

        std::string repr() const;

    private:
        const double R, kappa, B;
        const double Rin, Rout;

        double dzdr(double r) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Quadric& q);

}
#endif
