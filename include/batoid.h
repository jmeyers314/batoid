#ifndef batoid_batoid_h
#define batoid_batoid_h

#include <array>
#include "rayVector.h"
#include "surface.h"

namespace batoid {
    using vec3 = std::array<double, 3>;
    using mat3 = std::array<double, 9>;  // Column major rotation matrix.

    void applyForwardTransform(const vec3 dr, const mat3 drot, RayVector& rv);
    void applyReverseTransform(const vec3 dr, const mat3 drot, RayVector& rv);
    void intersect(const Surface& surface, const vec3 dr, const mat3 drot, RayVector& rv);
}

#endif
