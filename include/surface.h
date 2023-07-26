#ifndef batoid_surface_h
#define batoid_surface_h

#include "rayVector.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

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
            const double x, const double y, const double z,
            const double vx, const double vy, const double vz,
            double& dt, int niter
        ) const;
        virtual void grad(
            double x, double y,
            double& dzdx, double& dzdy
        ) const;

    protected:
        Surface();
        mutable Surface* _devPtr;

    private:
        #if defined(BATOID_GPU)
        void freeDevPtr() const;
        #endif
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

}
#endif
