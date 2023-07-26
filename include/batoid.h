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

    void applyForwardTransform(const vec3 dr, const mat3 drot, RayVector& rv, int max_threads);
    void applyReverseTransform(const vec3 dr, const mat3 drot, RayVector& rv, int max_threads);
    void obscure(const Obscuration& obsc, RayVector& rv, int max_threads);
    void intersect(
        const Surface& surface, const vec3 dr, const mat3 drot, RayVector& rv,
        const Coating* coating, int max_threads, int niter
    );
    void reflect(
        const Surface& surface, const vec3 dr, const mat3 drot, RayVector& rv,
        const Coating* coating, int max_threads, int niter
    );
    void refract(
        const Surface& surface, const vec3 dr, const mat3 drot,
        const Medium& m1, const Medium& m2, RayVector& rv, const Coating* coating,
        int max_threads, int niter
    );
    void rSplit(
        const Surface& surface, const vec3 dr, const mat3 drot,
        const Medium& m1, const Medium& m2,
        const Coating& coating,
        RayVector& rv, RayVector& rvSplit,
        int max_threads, int niter
    );
    void refractScreen(
        const Surface& surface, const vec3 dr, const mat3 drot,
        const Surface& screen, RayVector& rv,
        int max_threads, int niter
    );

    void applyForwardTransformArrays(
        const vec3 dr, const mat3 drot,
        double* x, double* y, double* z,
        size_t n, int max_threads
    );
    void applyReverseTransformArrays(
        const vec3 dr, const mat3 drot,
        double* x, double* y, double* z,
        size_t n, int max_threads
    );

    void finishParallel(const vec3 dr, const mat3 drot, const vec3 vv, double* x, double* y, double* z, size_t n);
}

#endif
