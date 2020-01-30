#ifndef batoid_quadric2_h
#define batoid_quadric2_h

#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"

namespace batoid {

    class Quadric2 : public Surface2CRTP<Quadric2> {
    public:
        Quadric2(double R, double conic);

        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

    private:
        const double _R;  // Radius of curvature
        const double _conic;  // Conic constant

        const double _Rsq;  // R*R
        const double _Rinvsq;  // 1/R/R
        const double _cp1; // 1 + conic
        const double _cp1inv; // 1/(1 + conic)
        const double _Rcp1; // R/(1+conic)
        const double _RRcp1cp1; // R*R/(1+conic)/(1+conic)
        const double _cp1RR; // (1+conic)/R/R

        double _dzdr(double r) const;
    };

}
#endif
