#ifndef __jem_paraboloid__h
#define __jem_paraboloid__h

#include "jtrace.h"

namespace jtrace {

    class Paraboloid : public Surface {
    public:
        Paraboloid(double _A, double _B);
        virtual double operator()(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Intersection intersect(const Ray&) const;
        double getA() const {return A;}
        double getB() const {return B;}

        std::string repr() const;

    private:
        double A, B;
    };

    inline std::ostream& operator<<(std::ostream& os, const Paraboloid &p);

}
#endif
