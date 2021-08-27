#ifndef batoid_batoid_h
#define batoid_batoid_h

#include <array>
#include "rayVector.h"
#include "surface.h"
#include "medium.h"
#include "obscuration.h"
#include "coating.h"

namespace batoid {
    using vec3 = std::array<double, 3>;
    using mat3 = std::array<double, 9>;  // Column major rotation matrix.

    void applyForwardTransform(const vec3 dr, const mat3 drot, RayVector& rv);
    void applyReverseTransform(const vec3 dr, const mat3 drot, RayVector& rv);
    void obscure(const Obscuration& obsc, RayVector& rv);
    void intersect(
        const Surface& surface, const vec3 dr, const mat3 drot, RayVector& rv,
        const Coating* coating
    );
    void reflect(
        const Surface& surface, const vec3 dr, const mat3 drot, RayVector& rv,
        const Coating* coating
    );
    void refract(
        const Surface& surface, const vec3 dr, const mat3 drot,
        const Medium& m1, const Medium& m2, RayVector& rv, const Coating* coating
    );
    void rSplit(
        const Surface& surface, const vec3 dr, const mat3 drot,
        const Medium& m1, const Medium& m2,
        const Coating& coating,
        RayVector& rv, RayVector& rvSplit
    );
    void refractScreen(
        const Surface& surface, const vec3 dr, const mat3 drot,
        const Surface& screen, RayVector& rv
    );

    void applyForwardTransformArrays(
        const vec3 dr, const mat3 drot,
        double* x, double* y, double* z,
        size_t n
    );
    void applyReverseTransformArrays(
        const vec3 dr, const mat3 drot,
        double* x, double* y, double* z,
        size_t n
    );

    void finishParallel(const vec3 dr, const mat3 drot, const vec3 vv, double* r, size_t n);
}

#endif
