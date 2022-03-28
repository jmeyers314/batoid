#ifndef batoid_surface_h
#define batoid_surface_h

#include "rayVector.h"

namespace batoid {

    /////////////
    // Surface //
    /////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Surface {
    public:
        virtual ~Surface();

        virtual double sag(double x, double y) const = 0;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const = 0;
        virtual bool timeToIntersect(
            const double x, const double y, const double z,
            const double vx, const double vy, const double vz,
            double& dt
        ) const;
        virtual void grad(
            double x, double y,
            double& dzdx, double& dzdy
        ) const;

    protected:
        Surface();
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////////
    // SurfaceHandle //
    ///////////////////

    class SurfaceHandle {
    public:
        SurfaceHandle();

        virtual ~SurfaceHandle();

        const Surface* getPtr() const;

        const Surface* getHostPtr() const;

    protected:
        Surface* _hostPtr;
        Surface* _devicePtr;
    };

}
#endif
