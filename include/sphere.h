#ifndef batoid_sphere_h
#define batoid_sphere_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Sphere : public Surface {
    public:
        Sphere(double R);
        virtual double sag(double, double) const override;
        virtual Vector3d normal(double, double) const override;
        bool timeToIntersect(const Ray& r, double& t) const override;

        double getR() const { return _R; }

    private:
        const double _R;  // Radius of curvature
        const double _Rsq; // R*R
        const double _Rinv; // 1/R
        const double _Rinvsq; // 1/R/R

        double dzdr(double r) const;
    };

}
#endif
