#include "surface.h"
#include "utils.h"

namespace batoid {
    Ray Surface::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.flux, r.vignetted);
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
        std::vector<Ray> rays(rv.size());

        parallelTransform(rv.cbegin(), rv.cend(), rays.begin(),
            [this](const Ray& ray)
            { return intersect(ray); }
        );
        return RayVector(std::move(rays), rv.getWavelength());
    }

    void Surface::intersectInPlace(RayVector& rv) const {
        parallel_for_each(
            rv.begin(), rv.end(),
            [this](Ray& r) { intersectInPlace(r); }
        );
    }

    // Note that I'm combining intersection and reflection/refraction here.
    Ray Surface::reflect(const Ray& r, const Coating* coating) const {
        if (r.failed) return r;
        Ray r2 = intersect(r);
        if (r2.failed) return r2;
        double n = 1.0 / r2.v.norm();
        Vector3d nv = r2.v * n;
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double alpha = nv.dot(normVec);

        if (coating)
            r2.flux *= coating->getReflect(r.wavelength, alpha);
        r2.v = (nv - 2*alpha*normVec).normalized()/n;
        return r2;
    }

    RayVector Surface::reflect(const RayVector& rv, const Coating* coating) const {
        std::vector<Ray> rv2(rv.size());
        parallelTransform(
            rv.cbegin(), rv.cend(), rv2.begin(),
            [this,coating](const Ray& r){ return reflect(r, coating); }
        );
        return RayVector(std::move(rv2), rv.getWavelength());
    }

    void Surface::reflectInPlace(Ray& r, const Coating* coating) const {
        if (r.failed) return;
        intersectInPlace(r);
        if (r.failed) return;
        double n = 1.0 / r.v.norm();
        Vector3d nv = r.v * n;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        double alpha = nv.dot(normVec);
        if (coating)
            r.flux *= coating->getReflect(r.wavelength, alpha);
        r.v = (nv - 2*alpha*normVec).normalized()/n;
    }

    void Surface::reflectInPlace(RayVector& rv, const Coating* coating) const {
        parallel_for_each(
            rv.begin(), rv.end(),
            [this,coating](Ray& r) { reflectInPlace(r, coating); }
        );
    }

    Ray Surface::refract(const Ray& r, const double n1, const double n2, const Coating* coating) const {
        if (r.failed) return r;
        Ray r2 = intersect(r);
        if (r2.failed) return r2;
        Vector3d nv = r2.v * n1;
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double alpha = nv.dot(normVec);
        if (coating)
            r2.flux *= coating->getTransmit(r.wavelength, alpha);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vector3d f1 = (nv+k1*normVec).normalized();
        Vector3d f2 = (nv+k2*normVec).normalized();
        r2.v = (f1.dot(nv) > f2.dot(nv)) ? f1/n2 : f2/n2;
        return r2;
    }

    Ray Surface::refract(const Ray& r, const Medium& m1, const Medium& m2, const Coating* coating) const {
        if (r.failed) return r;
        return refract(r, m1.getN(r.wavelength), m2.getN(r.wavelength), coating);
    }

    RayVector Surface::refract(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating* coating) const {
        std::vector<Ray> rays(rv.size());

        // use double version of refract if possible
        if (std::isnan(rv.getWavelength())) {
            parallelTransform(
                rv.cbegin(), rv.cend(), rays.begin(),
                [this,&m1,&m2,coating](const Ray& r){ return refract(r, m1, m2, coating); }
            );
        } else {
            double n1 = m1.getN(rv.getWavelength());
            double n2 = m2.getN(rv.getWavelength());
            parallelTransform(
                rv.cbegin(), rv.cend(), rays.begin(),
                [this,n1,n2,coating](const Ray& r){ return refract(r, n1, n2, coating); }
            );
        }
        return RayVector(std::move(rays), rv.getWavelength());
    }

    void Surface::refractInPlace(Ray& r, const double n1, const double n2, const Coating* coating) const {
        if (r.failed) return;
        intersectInPlace(r);
        if (r.failed) return;
        Vector3d nv = r.v * n1;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        double alpha = nv.dot(normVec);
        if (coating)
            r.flux *= coating->getTransmit(r.wavelength, alpha);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vector3d f1 = (nv+k1*normVec).normalized();
        Vector3d f2 = (nv+k2*normVec).normalized();
        r.v = (f1.dot(nv) > f2.dot(nv)) ? f1/n2 : f2/n2;
    }

    void Surface::refractInPlace(Ray& r, const Medium& m1, const Medium& m2, const Coating* coating) const {
        if (r.failed) return;
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        refractInPlace(r, n1, n2, coating);
    }

    void Surface::refractInPlace(RayVector& rv, const Medium& m1, const Medium& m2, const Coating* coating) const {
        // Use double version of refractInPlace if possible
        if (std::isnan(rv.getWavelength())) {
            parallel_for_each(
                rv.begin(), rv.end(),
                [this,&m1,&m2,coating](Ray& r){ refractInPlace(r, m1, m2, coating); }
            );
        } else {
            double n1 = m1.getN(rv.getWavelength());
            double n2 = m2.getN(rv.getWavelength());
            parallel_for_each(
                rv.begin(), rv.end(),
                [this,n1,n2,coating](Ray& r){ refractInPlace(r, n1, n2, coating); }
            );
        }
    }

    std::pair<Ray, Ray> Surface::rSplit(const Ray& r, const double n1, const double n2, const Coating& coating) const {
        if (r.failed) return std::make_pair(r, r);
        Ray r2 = intersect(r);
        if (r2.failed) return std::make_pair(r2, r2);

        // Common calculations
        Vector3d nv = r.v * n1;  // Makes this a unit vector...
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double alpha = nv.dot(normVec);

        // Flux coefficients
        double reflect, transmit;
        coating.getCoefs(r.wavelength, alpha, reflect, transmit);

        // Reflection calculation
        Ray reflectedRay(
            r2.r, (nv - 2*alpha*normVec).normalized()/n1,
            r2.t, r2.wavelength, reflect*r2.flux, r2.vignetted
        );

        // Refraction calculation
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vector3d f1 = (nv+k1*normVec).normalized();
        Vector3d f2 = (nv+k2*normVec).normalized();
        // Use r2 instead of creating a new Ray
        r2.v = f1.dot(nv)>f2.dot(nv) ? f1/n2 : f2/n2;
        r2.flux = transmit*r2.flux;
        return std::make_pair(reflectedRay, r2);
    }

    std::pair<Ray, Ray> Surface::rSplit(const Ray& r, const Medium& m1, const Medium& m2, const Coating& coating) const {
        return rSplit(r, m1.getN(r.wavelength), m2.getN(r.wavelength), coating);
    }

    std::pair<RayVector, RayVector> Surface::rSplit(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating& coating) const {
        RayVector reflected(rv);
        RayVector refracted(rv);

        reflectInPlace(reflected, &coating);
        refractInPlace(refracted, m1, m2, &coating);

        return std::make_pair(reflected, refracted);
    }
}
