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
    void obscure(const ObscurationHandle& obsc, RayVector& rv);
    void intersect(
        const SurfaceHandle& surface, const vec3 dr, const mat3 drot, RayVector& rv,
        const CoatingHandle* coating
    );
    void reflect(
        const SurfaceHandle& surface, const vec3 dr, const mat3 drot, RayVector& rv,
        const CoatingHandle* coating
    );
    void refract(
        const SurfaceHandle& surface, const vec3 dr, const mat3 drot,
        const MediumHandle& m1, const MediumHandle& m2, RayVector& rv, const CoatingHandle* coating
    );
    void rSplit(
        const SurfaceHandle& surface, const vec3 dr, const mat3 drot,
        const MediumHandle& m1, const MediumHandle& m2,
        const CoatingHandle& coating,
        RayVector& rv, RayVector& rvSplit
    );
    void refractScreen(
        const SurfaceHandle& surface, const vec3 dr, const mat3 drot,
        const SurfaceHandle& screen, RayVector& rv
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

    void finishParallel(const vec3 dr, const mat3 drot, const vec3 vv, double* x, double* y, double* z, size_t n);
}

#endif
