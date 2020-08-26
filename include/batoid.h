#ifndef batoid_batoid_h
#define batoid_batoid_h

#include "rayVector.h"
#include "surface.h"
#include "coordTransform.h"
// #include "medium.h"
// #include "utils.h"

namespace batoid {
    void intersect(const Surface& surface, RayVector& rv, CoordTransform* ct=nullptr);
}

#endif
