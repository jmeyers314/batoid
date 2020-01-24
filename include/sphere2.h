#ifndef batoid_sphere2_h
#define batoid_sphere2_h

#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"

namespace batoid {

    class Sphere2 : public Surface2CRTP<Sphere2> {
    public:
        Sphere2(double R);

        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

        // void _intersectInPlace(RayVector2&) const;
        // void _reflectInPlace(RayVector2&) const;
        // void _refractInPlace(RayVector2&, const Medium2&, const Medium2&) const;

    private:
        const double _R;  // Radius of curvature
        const double _Rsq; // R*R
        const double _Rinv; // 1/R
        const double _Rinvsq; // 1/R/R

        double _dzdr(double r) const;
    };
}
#endif
