#ifndef jtrace_plane_h
#define jtrace_plane_h

#include "jtrace.h"

namespace jtrace {

    class Plane : public Surface {
    public:
        Plane(double);
        virtual double operator()(double, double) const {return B;}
        virtual Vec3 normal(double, double) const {return Vec3(0,0,1);}
        virtual Intersection intersect(const Ray&) const;
        double getB() const {return B;}

        std::string repr() const;

    private:
        const double B;
    };

    inline std::ostream& operator<<(std::ostream& os, const Plane& p);
}
#endif
