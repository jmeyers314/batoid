#ifndef batoid_surface_h
#define batoid_surface_h

#include "rayVector.h"

namespace batoid {
    class Surface {
    public:
        virtual ~Surface();

        virtual const Surface* getDevPtr() const = 0;

        virtual double sag(double x, double y) const = 0;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const = 0;
        virtual bool timeToIntersect(
            double x, double y, double z,
            double vx, double vy, double vz,
            double& dt
        ) const;
        virtual void grad(
            double x, double y,
            double& dzdx, double& dzdy
        ) const;

    protected:
        Surface();
        mutable Surface* _devPtr;
    };
}
#endif
