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
        virtual double sag(double, double) const;
        virtual Vector3d normal(double, double) const;
        Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        double getR() const {return _R;}
        double getConic() const {return _conic;}
        std::string repr() const;


    protected:
        bool timeToIntersect(const Ray& r, double& t) const;
        double dzdr(double r) const;

    private:
        const double _R;  // Radius of curvature
        const double _conic;  // Conic constant

        const double _Rsq;  // R*R
        const double _Rinvsq;  // 1/R/R
        const double _cp1; // 1 + conic
        const double _cp1inv; // 1/(1 + conic)
        const double _Rcp1; // R/(1+conic)
        const double _RRcp1cp1; // R*R/(1+conic)/(1+conic)
        const double _cp1RR; // (1+conic)/R/R
    };

    inline bool operator==(const Quadric& q1, const Quadric& q2)
        { return q1.getR() == q2.getR() && q1.getConic() == q2.getConic(); }
    inline bool operator!=(const Quadric& q1, const Quadric& q2)
        { return q1.getR() != q2.getR() || q1.getConic() != q2.getConic(); }

}
#endif
