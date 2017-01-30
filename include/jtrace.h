#ifndef __jem_jtrace__h
#define __jem_jtrace__h

#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include "vec3.h"
#include "ray.h"
#include "intersection.h"
#include "surface.h"
#include "paraboloid.h"

namespace jtrace {

    class NoIntersectionError
    {
    public:
        explicit NoIntersectionError(const char *_message) : message(_message) {}
        const char *GetMessage() const {return message;}
    private:
        const char * const message;
    };

    class NoFutureIntersectionError
    {
    public:
        explicit NoFutureIntersectionError(const char *_message) : message(_message) {}
        const char *GetMessage() const {return message;}
    private:
        const char * const message;
    };

    // class Asphere : public Surface {
    // public:
    //     Asphere(double _R, double _kappa, std::vector<double> _alpha, double _B) :
    //         R(_R), kappa(_kappa), alpha(_alpha), B(_B) {};
    //     virtual double operator()(double, double) const;
    //     virtual Vec3 normal(double, double) const;
    //     virtual Intersection intersect(const Ray&) const;
    // private:
    //     double R, kappa;
    //     std::vector<double> alpha;
    //     double B;
    //
    //     double dzdr(double r) const;
    // };
} // namespace jtrace

#endif
