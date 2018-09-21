#include "surface.h"
#include "utils.h"

namespace batoid {
    Ray Surface::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.vignetted);
    }

    void Surface::intersectInPlace(Ray& r) const {
        if (r.failed) return;
        double t;
        if (!timeToIntersect(r, t)) {
            r.failed=true;
            r.vignetted=true; // All failed rays are considered vignetted
            return;
        }
        r.r = r.positionAtTime(t);
        r.t = t;
        return;
    }

    RayVector Surface::intersect(const RayVector& rv) const {
        std::vector<Ray> rays(rv.rays.size());

        parallelTransform(rv.rays.cbegin(), rv.rays.cend(), rays.begin(),
            [this](const Ray& ray)
            { return intersect(ray); }
        );
        return RayVector(std::move(rays), rv.wavelength);
    }

    void Surface::intersectInPlace(RayVector& rv) const {
        parallel_for_each(
            rv.rays.begin(), rv.rays.end(),
            [this](Ray& r) { intersectInPlace(r); }
        );
    }

    // Note that I'm combining intersection and reflection/refraction here.
    Ray Surface::reflect(const Ray& r) const {
        if (r.failed) return r;
        Ray r2 = intersect(r);
        if (r2.failed) return r2;
        double n = 1.0 / r2.v.norm();
        Vector3d nv = r2.v * n;
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double c1 = nv.dot(normVec);
        return Ray(r2.r, (nv - 2*c1*normVec).normalized()/n, r2.t, r2.wavelength, r2.vignetted);
    }

    RayVector Surface::reflect(const RayVector& rv) const {
        std::vector<Ray> rv2(rv.rays.size());
        parallelTransform(
            rv.rays.cbegin(), rv.rays.cend(), rv2.begin(),
            [this](const Ray& r){ return reflect(r); }
        );
        return RayVector(std::move(rv2), rv.wavelength);
    }

    void Surface::reflectInPlace(Ray& r) const {
        if (r.failed) return;
        intersectInPlace(r);
        if (r.failed) return;
        double n = 1.0 / r.v.norm();
        Vector3d nv = r.v * n;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        double c1 = nv.dot(normVec);
        r.v = (nv - 2*c1*normVec).normalized()/n;
    }

    void Surface::reflectInPlace(RayVector& rv) const {
        parallel_for_each(
            rv.rays.begin(), rv.rays.end(),
            [this](Ray& r) { reflectInPlace(r); }
        );
    }

    Ray Surface::refract(const Ray& r, const double n1, const double n2) const {
        if (r.failed) return r;
        Ray r2 = intersect(r);
        if (r2.failed) return r2;
        Vector3d nv = r2.v * n1;
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double alpha = nv.dot(normVec);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vector3d f1 = (nv+k1*normVec).normalized();
        Vector3d f2 = (nv+k2*normVec).normalized();
        if (f1.dot(nv) > f2.dot(nv))
            return Ray(r2.r, f1/n2, r2.t, r2.wavelength, r2.vignetted);
        else
            return Ray(r2.r, f2/n2, r2.t, r2.wavelength, r2.vignetted);
    }

    Ray Surface::refract(const Ray& r, const Medium& m1, const Medium& m2) const {
        if (r.failed) return r;
        return refract(r, m1.getN(r.wavelength), m2.getN(r.wavelength));
    }

    RayVector Surface::refract(const RayVector& rv, const Medium& m1, const Medium& m2) const {
        std::vector<Ray> rays(rv.rays.size());

        // use double version of refract if possible
        if (std::isnan(rv.wavelength)) {
            parallelTransform(
                rv.rays.cbegin(), rv.rays.cend(), rays.begin(),
                [this,&m1,&m2](const Ray& r){ return refract(r, m1, m2); }
            );
        } else {
            double n1 = m1.getN(rv.wavelength);
            double n2 = m2.getN(rv.wavelength);
            parallelTransform(
                rv.rays.cbegin(), rv.rays.cend(), rays.begin(),
                [this,n1,n2](const Ray& r){ return refract(r, n1, n2); }
            );
        }
        return RayVector(std::move(rays), rv.wavelength);
    }

    void Surface::refractInPlace(Ray& r, const double n1, const double n2) const {
        if (r.failed) return;
        intersectInPlace(r);
        if (r.failed) return;
        Vector3d nv = r.v * n1;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        double alpha = nv.dot(normVec);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vector3d f1 = (nv+k1*normVec).normalized();
        Vector3d f2 = (nv+k2*normVec).normalized();
        if (f1.dot(nv) > f2.dot(nv))
            r.v = f1/n2;
        else
            r.v = f2/n2;
    }

    void Surface::refractInPlace(Ray& r, const Medium& m1, const Medium& m2) const {
        if (r.failed) return;
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        refractInPlace(r, n1, n2);
    }

    void Surface::refractInPlace(RayVector& rv, const Medium& m1, const Medium& m2) const {
        // Use double version of refractInPlace if possible
        if (std::isnan(rv.wavelength)) {
            parallel_for_each(
                rv.rays.begin(), rv.rays.end(),
                [this, &m1, &m2](Ray& r){ refractInPlace(r, m1, m2); }
            );
        } else {
            double n1 = m1.getN(rv.wavelength);
            double n2 = m2.getN(rv.wavelength);
            parallel_for_each(
                rv.rays.begin(), rv.rays.end(),
                [this, n1, n2](Ray& r){ refractInPlace(r, n1, n2); }
            );
        }
    }
}
