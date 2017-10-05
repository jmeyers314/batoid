#include "batoid.h"
#include "ray.h"
#include "surface.h"
#include "medium.h"
#include "utils.h"
#include "intersection.h"

namespace batoid{
    Ray reflect(const Ray& r, const Surface& surface) {
        if (r.failed)
            return Ray(true);
        double n = 1.0 / r.v.Magnitude();
        Vec3 nv = r.v * n;
        Vec3 normal = surface.normal(r.p0.x, r.p0.y);
        double c1 = DotProduct(nv, normal);
        return Ray(r.p0, (nv - 2*c1*normal).UnitVec3()/n, r.t0, r.wavelength, r.isVignetted);
    }

    std::vector<Ray> reflect(const std::vector<Ray>& rays, const Surface& surface) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return reflect(r, surface); },
            2000
        );
        return result;
    }

    Ray refract(const Ray& r, const Surface& surface, const double n1, const double n2) {
        if (r.failed)
            return Ray(true);
        Vec3 nv = r.v * n1;
        Vec3 normal = surface.normal(r.p0.x, r.p0.y);
        double alpha = DotProduct(nv, normal);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vec3 f1 = (nv+k1*normal).UnitVec3();
        Vec3 f2 = (nv+k2*normal).UnitVec3();
        if (DotProduct(f1, nv) > DotProduct(f2, nv))
            return Ray(r.p0, f1/n2, r.t0, r.wavelength, r.isVignetted);
        else
            return Ray(r.p0, f2/n2, r.t0, r.wavelength, r.isVignetted);

    }

    std::vector<Ray> refract(const std::vector<Ray>& rays, const Surface& surface,
                             const double n1, const double n2) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return refract(r, surface, n1, n2); },
            2000
        );
        return result;
    }

    Ray refract(const Ray& r, const Surface& surface, const Medium& m1, const Medium& m2) {
        if (r.failed)
            return Ray(true);
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        return refract(r, surface, n1, n2);
    }

    std::vector<Ray> refract(const std::vector<Ray>& rays, const Surface& surface,
                             const Medium& m1, const Medium& m2) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return refract(r, surface, m1, m2); },
            2000
        );
        return result;
    }

}
