#ifndef __jem_asphere__h
#define __jem_asphere__h

#include "jtrace.h"
#include "surface.h"
#include "solve.h"

namespace jtrace {

    class Asphere : public Surface {
    public:
        Asphere(double _R, double _kappa, std::vector<double> _alpha, double _B);
        virtual double operator()(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Intersection intersect(const Ray&) const;
        double getR() const {return R;}
        double getKappa() const {return kappa;}
        const std::vector<double>& getAlpha() const {return alpha;}
        double getB() const {return B;}

        std::string repr() const;

    private:
        double R, kappa;
        std::vector<double> alpha;
        double B;

        double dzdr(double r) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Asphere &a);

}
#endif
