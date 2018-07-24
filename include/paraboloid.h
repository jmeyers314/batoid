#ifndef batoid_paraboloid_h
#define batoid_paraboloid_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Paraboloid : public Surface {
    public:
        Paraboloid(double R);
        virtual double sag(double, double) const override;
        virtual Vector3d normal(double, double) const override;
        virtual bool operator==(const Surface& rhs) const override;
        bool timeToIntersect(const Ray& r, double& t) const override;

        double getR() const {return _R;}
        std::string repr() const override;

    private:
        const double _R;  // Radius of curvature
        const double _Rinv;  // 1/R
        const double _2Rinv;  // 1/(2*R)
    };

}
#endif
