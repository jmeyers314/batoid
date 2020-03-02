#ifndef batoid_bicubic2_h
#define batoid_bicubic2_h

#include "surface2.h"
#include "rayVector2.h"
#include "dualView.h"

namespace batoid {
    class Bicubic2 : public Surface2CRTP<Bicubic2> {
    public:
        Bicubic2(
            double x0, double y0, double dx, double dy,
            double* z, double* dzdx, double* dzdy, double*d2zdxdy,
            size_t size
        );
        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

    private:
        double oneDSpline(double x, double val0, double val1, double der0, double der1) const;
        double oneDGrad(double x, double val0, double val1, double der0, double der1) const;

        double _x0, _y0;
        double _dx, _dy;
        DualView<double> _z;
        DualView<double> _dzdx;
        DualView<double> _dzdy;
        DualView<double> _d2zdxdy;
        size_t _size;
    };
}
#endif
