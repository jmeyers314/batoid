#ifndef jtrace_quadric_h
#define jtrace_quadric_h

#include "jtrace.h"

namespace jtrace {

    class Quadric : public Surface {
    public:
        Quadric(double _R, double _kappa, double _B);
        virtual double operator()(double, double) const;
        virtual Vec3 normal(double, double) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        double getR() const {return R;}
        double getKappa() const {return kappa;}
        double getB() const {return B;}

        std::string repr() const;

    private:
        const double R, kappa, B;

        double dzdr(double r) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Quadric& q);

}
#endif
