#ifndef batoid_asphere2_h
#define batoid_asphere2_h

#include <array>
#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"
#include "quadric2.h"

namespace batoid {

    class Asphere2 : public Surface2CRTP<Asphere2> {
    public:
        Asphere2(double R, double conic, double* coefs, size_t size);
        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

    private:
        DualView<double> _coefs;
        DualView<double> _dzdrcoefs;
        size_t _size;
        const Quadric2 _q;

        double _dzdr(double r) const;
        void _computeDzDrCoefs();
    };

}
#endif
