#ifndef batoid_paraboloid2_h
#define batoid_paraboloid2_h

#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"

namespace batoid {

    class Paraboloid2 : public Surface2CRTP<Paraboloid2> {
    public:
        Paraboloid2(double R);

        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

    private:
        const double _R;  // Radius of curvature
        const double _Rinv; // 1/R
        const double _2Rinv; // 1/(2*R)
    };
}
#endif
