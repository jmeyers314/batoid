#ifndef batoid_paraboloid_h
#define batoid_paraboloid_h

#include "surface.h"

namespace batoid {

    class Paraboloid : public Surface {
    public:
        Paraboloid(double R);
        ~Paraboloid();

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
        const double _Rinv;  // 1/R
        const double _2Rinv;  // 1/(2*R)
    };

}
#endif
