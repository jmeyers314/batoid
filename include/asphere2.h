#ifndef batoid_asphere_h
#define batoid_asphere_h

#include <array>
#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"
#include "quadric2.h"

namespace batoid {

    class Asphere2 : public Surface2CRTP<Asphere2> {
    public:
        Asphere2(double R, double conic, const std::array<double, 10>& coefs);
        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

    private:
        const std::array<double, 10> _coefs;  // allocate on stack to avoid dynamic host<->device transfers.
        const std::array<double, 10> _dzdrcoefs;
        const Quadric2 _q;

        double _dzdr(double r) const;
        static std::array<double,10> _computeDzDrCoefs(const std::array<double, 10>& coefs);
    };

}
#endif
