#ifndef batoid_sum_h
#define batoid_sum_h

#include <vector>
#include "surface.h"
#include "ray.h"

using Eigen::Vector3d;

namespace batoid {

    class Sum : public Surface {
    public:
        Sum(std::vector<std::shared_ptr<Surface>> surfaces);

        virtual double sag(double, double) const;
        virtual Vector3d normal(double, double) const;
        virtual Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        const std::vector<std::shared_ptr<Surface>>& getSurfaces() const { return _surfaces; }
        std::string repr() const;

    private:
        const std::vector<std::shared_ptr<Surface>> _surfaces;

        bool timeToIntersect(const Ray& r, double& t) const;
    };
}

#endif
