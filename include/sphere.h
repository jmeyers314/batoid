#ifndef batoid_sphere_h
#define batoid_sphere_h

#include "surface.h"

namespace batoid {

    class Sphere : public Surface {
    public:
        Sphere(double R);
        ~Sphere();

        virtual const Surface* getDevPtr() const override;

        virtual double sag(double, double) const override;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const override;
        virtual bool timeToIntersect(
            double x, double y, double z,
            double vx, double vy, double vz,
            double& dt
        ) const override;

    private:
        const double _R;  // Radius of curvature
        const double _Rsq; // R*R
        const double _Rinv; // 1/R
        const double _Rinvsq; // 1/R/R

        double _dzdr(double r) const;
    };

}
#endif
