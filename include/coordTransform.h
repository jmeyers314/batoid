#ifndef batoid_coordTransform_h
#define batoid_coordTransform_h

#include "rayVector.h"
#include "coordSys.h"

namespace batoid {
    struct CoordTransform {
    public:
        CoordTransform(const CoordSys& _source, const CoordSys& _destination);

        void applyForwardInPlace(RayVector& rv) const;
        void applyReverseInPlace(RayVector& rv) const;

        const vec3 dr;
        const mat3 rot;
        const CoordSys source;
        const CoordSys destination;
    };
}

#endif
