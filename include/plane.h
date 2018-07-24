#ifndef batoid_plane_h
#define batoid_plane_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Plane : public Surface {
    public:
        virtual double sag(double, double) const override { return 0.0; }
        virtual Vector3d normal(double, double) const override { return Vector3d(0.0, 0.0, 1.0); }
        virtual bool operator==(const Surface& rhs) const override;
        bool timeToIntersect(const Ray& r, double& t) const override;

        std::string repr() const override;
    };
}
#endif
