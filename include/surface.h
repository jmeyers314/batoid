#ifndef batoid_surface_h
#define batoid_surface_h

#include <vector>
#include <memory>
#include "ray.h"
#include "rayVector.h"
#include "medium.h"

#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    class Surface {
    public:
        virtual double sag(double, double) const = 0;
        virtual Vector3d normal(double, double) const = 0;

        virtual Ray intersect(const Ray&) const = 0;
        virtual void intersectInPlace(Ray&) const = 0;

        RayVector intersect(const RayVector&) const;
        void intersectInPlace(RayVector&) const;

        Ray reflect(const Ray&) const;
        RayVector reflect(const RayVector&) const;

        void reflectInPlace(Ray&) const;
        void reflectInPlace(RayVector&) const;

        Ray refract(const Ray&, const Medium&, const Medium&) const;
        RayVector refract(const RayVector&, const Medium&, const Medium&) const;

        void refractInPlace(Ray&, const Medium&, const Medium&) const;
        void refractInPlace(RayVector&, const Medium&, const Medium&) const;

        virtual std::string repr() const = 0;
    private:
        Ray refract(const Ray&, const double, const double) const;
        void refractInPlace(Ray&, const double, const double) const;
    };
    std::ostream& operator<<(std::ostream& os, const Surface& s);
}
#endif
