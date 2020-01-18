#ifndef batoid_plane2_h
#define batoid_plane2_h

#include <sstream>
#include <limits>
#include "surface2.h"
#include "rayVector2.h"

namespace batoid {

    class Plane2 : public Surface2 {
    public:
        Plane2(bool allowReverse=false) : _allowReverse(allowReverse) {}
        virtual void intersectInPlace(RayVector2&) const override;
        virtual void reflectInPlace(RayVector2&) const override;
        virtual void refractInPlace(RayVector2&, const Medium2&, const Medium2&) const override;

        bool getAllowReverse() const {return _allowReverse;}

    private:
        bool _allowReverse;
    };
}
#endif
