#ifndef batoid_bilinear2_h
#define batoid_bilinear2_h

#include <array>
#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"
#include "dualView.h"

namespace batoid {

    class Bilinear2 : public Surface2CRTP<Bilinear2> {
    public:
        Bilinear2(double x0, double y0, double dx, double dy, double* z, size_t size);
        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

    private:
        double _x0, _y0;
        double _dx, _dy;
        DualView<double> _z;
        size_t _size;
    };

}
#endif
