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
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double alpha = r2.v.dot(normVec);
        r2.v -= 2*alpha*normVec;
        if (coating)
            r2.flux *= coating->getReflect(r.wavelength, alpha/r2.v.norm());
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
        Vector3d normVec(normal(r.r[0], r.r[1]));
        double alpha = r.v.dot(normVec);
        r.v -= 2*alpha*normVec;
        if (coating)
            r.flux *= coating->getReflect(r.wavelength, alpha/r.v.norm());
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
        Vector3d i = r2.v * n1;
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double cos = i.dot(normVec);
        if (cos > 0.) {
            normVec *= -1;
            cos *= -1;
        }
        double eta = n1/n2;
        double sinsqr = eta*eta*(1-cos*cos);
        Vector3d t = eta * i - (eta * cos + std::sqrt(1 - sinsqr)) * normVec;
        r2.v = t/n2;
        if (coating)
            r2.flux *= coating->getTransmit(r2.wavelength, cos);
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
        Vector3d i = r.v*n1;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        double cos = i.dot(normVec);
        if (cos > 0.) {
            normVec *= -1;
            cos *= -1;
        }
        double eta = n1/n2;
        double sinsqr = eta*eta*(1-cos*cos);
        Vector3d t = eta * i - (eta * cos + std::sqrt(1 - sinsqr)) * normVec;
        r.v = t/n2;
        if (coating)
            r.flux *= coating->getTransmit(r.wavelength, cos);
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
        Vector3d i = r.v * n1;
        Vector3d normVec(normal(r2.r[0], r2.r[1]));
        double cos = i.dot(normVec);
        if (cos > 0.) {
            normVec *= -1.;
            cos *= -1;
        }

        // Flux coefficients
        double reflect, transmit;
        coating.getCoefs(r.wavelength, cos, reflect, transmit);

        // Reflection calculation
        Ray reflectedRay(
            r2.r, (i - 2*cos*normVec)/n1,
            r2.t, r2.wavelength, reflect*r2.flux, r2.vignetted
        );

        // Refraction calculation
        double eta = n1/n2;
        double sinsqr = eta*eta*(1-cos*cos);
        Vector3d t = eta * i - (eta * cos + std::sqrt(1 - sinsqr)) * normVec;
        r2.v = t/n2;
        r2.flux *= transmit;
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

    std::pair<RayVector, RayVector> Surface::rSplitProb(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating& coating) const {
        RayVector reflected(rv);
        RayVector refracted(rv);

        reflectInPlace(reflected);
        refractInPlace(refracted, m1, m2);

        // Go through and probabilistically accept/reject each ray?
        double reflect, transmit, alpha, ran;
        for(unsigned int i=0; i<rv.size(); i++) {
            // Need to recompute the normal vector and alpha=cos(theta)...  for the third time...
            Vector3d normVec(normal(rv[i].r[0], rv[i].r[1]));
            alpha = rv[i].v.normalized().dot(normVec);
            coating.getCoefs(rv[i].wavelength, alpha, reflect, transmit);
            ran = std::uniform_real_distribution<>(0.0, 1.0)(rng);
            if (ran < reflect) { //choose reflect
                refracted[i].vignetted=true;
            } else if (ran < reflect+transmit) { // choose refract
                reflected[i].vignetted=true;
            } else { // choose neither
                refracted[i].vignetted=true;
                reflected[i].vignetted=true;
            }
        }
        reflected.trimVignettedInPlace(0.0);
        refracted.trimVignettedInPlace(0.0);
        return std::make_pair(reflected, refracted);
    }
}
