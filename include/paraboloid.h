#ifndef __jem_paraboloid__h
#define __jem_paraboloid__h

#include "jtrace.h"
#include "surface.h"

namespace jtrace {

    class Paraboloid : public Surface {
    public:
        Paraboloid(double _A, double _B);
        // Paraboloid(double _A, double _B) : A(_A), B(_B) {}
        virtual double operator()(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Intersection intersect(const Ray&) const;
        double getA() const {return A;}
        double getB() const {return B;}

        std::string repr() const;
        // std::string repr() const {
        //     std::ostringstream oss(" ");
        //     oss << "Paraboloid(" << A << ", " << B << ")";
        //     return oss.str();
        // }
    private:
        double A, B;
    };

    inline std::ostream& operator<<(std::ostream& os, const Paraboloid &p);

}
#endif
