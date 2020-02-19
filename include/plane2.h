#ifndef batoid_plane2_h
#define batoid_plane2_h

#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"

namespace batoid {

    class Plane2 : public Surface2CRTP<Plane2> {
    public:
        Plane2(bool allowReverse=false) : _allowReverse(allowReverse) {}

        double _sag(double, double) const;
        void _normal(double, double, double&, double&, double&) const;
        bool _timeToIntersect(double, double, double, double, double, double, double&) const;

        void _intersectInPlace(RayVector2&, const CoordSys* cs=nullptr) const;
        void _reflectInPlace(RayVector2&, const CoordSys* cs=nullptr) const;
        void _refractInPlace(RayVector2&, const Medium2&, const Medium2&, const CoordSys* cs=nullptr) const;

        bool getAllowReverse() const {return _allowReverse;}

    private:
        bool _allowReverse;
    };
}
#endif
