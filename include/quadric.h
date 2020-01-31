#ifndef batoid_quadric_h
#define batoid_quadric_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Quadric : public Surface {
    public:
        Quadric(double R, double conic);
        virtual double sag(double, double) const override;
        virtual Vector3d normal(double, double) const override;
        virtual bool timeToIntersect(const Ray& r, double& t) const override;

        double getR() const {return _R;}
        double getConic() const {return _conic;}

    protected:
        const double _R;  // Radius of curvature
        const double _conic;  // Conic constant

        double dzdr(double r) const;

    private:
        const double _Rsq;  // R*R
        const double _Rinvsq;  // 1/R/R
        const double _cp1; // 1 + conic
        const double _cp1inv; // 1/(1 + conic)
        const double _Rcp1; // R/(1+conic)
        const double _RRcp1cp1; // R*R/(1+conic)/(1+conic)
        const double _cp1RR; // (1+conic)/R/R
    };

}
#endif
