#ifndef batoid_plane_h
#define batoid_plane_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include "rayVector4.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Plane : public Surface {
    public:
        Plane(bool allowReverse=false) : _allowReverse(allowReverse) {}
        virtual double sag(double, double) const override { return 0.0; }
        virtual Vector3d normal(double, double) const override { return Vector3d(0.0, 0.0, 1.0); }
        bool timeToIntersect(const Ray& r, double& t) const override;

        virtual void intersectInPlace(RayVector4&) const override;
        virtual void reflectInPlace(RayVector4&, const Coating* coating=nullptr) const override;
        virtual void refractInPlace(RayVector4&, const Medium&, const Medium&, const Coating* coating=nullptr) const override;

        bool getAllowReverse() const {return _allowReverse;}

    private:
        bool _allowReverse;
    };
}
#endif
