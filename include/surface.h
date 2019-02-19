#ifndef batoid_surface_h
#define batoid_surface_h

#include <vector>
#include <memory>
#include <utility>
#include "ray.h"
#include "rayVector.h"
#include "medium.h"
#include "coating.h"

#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    class Surface {
    public:
        virtual ~Surface() {}

        virtual double sag(double, double) const = 0;
        virtual Vector3d normal(double, double) const = 0;
        virtual bool timeToIntersect(const Ray& r, double& t) const = 0;

        Ray intersect(const Ray&) const;
        RayVector intersect(const RayVector&) const;
        void intersectInPlace(Ray&) const;
        void intersectInPlace(RayVector&) const;

        Ray reflect(const Ray&, const Coating* coating=nullptr) const;
        RayVector reflect(const RayVector&, const Coating* coating=nullptr) const;
        void reflectInPlace(Ray&, const Coating* coating=nullptr) const;
        void reflectInPlace(RayVector&, const Coating* coating=nullptr) const;

        Ray refract(const Ray&, const Medium&, const Medium&, const Coating* coating=nullptr) const;
        RayVector refract(const RayVector&, const Medium&, const Medium&, const Coating* coating=nullptr) const;
        void refractInPlace(Ray&, const Medium&, const Medium&, const Coating* coating=nullptr) const;
        void refractInPlace(RayVector&, const Medium&, const Medium&, const Coating* coating=nullptr) const;

        std::pair<Ray, Ray> rSplit(const Ray&, const Medium&, const Medium&, const Coating&) const;
        std::pair<RayVector, RayVector> rSplit(const RayVector&, const Medium&, const Medium&, const Coating&) const;
        std::pair<RayVector, RayVector> rSplitProb(const RayVector&, const Medium&, const Medium&, const Coating&) const;

    private:
        Ray refract(const Ray&, const double, const double, const Coating* coating=nullptr) const;
        void refractInPlace(Ray&, const double, const double, const Coating* coating=nullptr) const;

        std::pair<Ray, Ray> rSplit(const Ray&, const double, const double, const Coating&) const;
    };
}
#endif
