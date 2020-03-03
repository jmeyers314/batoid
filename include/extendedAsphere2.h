#ifndef batoid_extendedAsphere2_h
#define batoid_extendedAsphere2_h

#include <array>
#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"
#include "quadric2.h"

namespace batoid {

    class ExtendedAsphere2 : public Surface2CRTP<ExtendedAsphere2> {
    public:
        ExtendedAsphere2(
            double R, double conic, double* coefs, size_t ncoefs,
            double x0, double y0, double dx, double dy,
            double* z, double* dzdx, double* dzdy, double* d2zdxdy,
            size_t gridsize
        );
        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

    private:
        // Asphere part
        DualView<double> _coefs;
        DualView<double> _dzdrcoefs;
        size_t _ncoefs;
        const Quadric2 _q;

        // Bicubic part
        double _x0, _y0;
        double _dx, _dy;
        DualView<double> _z;
        DualView<double> _dzdx;
        DualView<double> _dzdy;
        DualView<double> _d2zdxdy;
        size_t _gridsize;

        double oneDSpline(double x, double val0, double val1, double der0, double der1) const;
        double oneDGrad(double x, double val0, double val1, double der0, double der1) const;

        double _dzdr(double r) const;
        void _computeDzDrCoefs();
    };

}
#endif
