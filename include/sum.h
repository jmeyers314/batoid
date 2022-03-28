#ifndef batoid_sum_h
#define batoid_sum_h

#include "surface.h"

namespace batoid {

    /////////
    // Sum //
    /////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Sum : public Surface {
    public:
        Sum(const Surface** surfaces, size_t nsurf);
        ~Sum();

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
        const Surface** _surfaces;
        size_t _nsurf;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////
    // SumHandle //
    ///////////////

    class SumHandle : public SurfaceHandle {
    public:
        SumHandle(const SurfaceHandle** handles, const size_t nsurf);
        virtual ~SumHandle();

        static const Surface** _getSurfaces(
            const SurfaceHandle** handles, const size_t nsurf, bool host
        );
    private:
        const Surface** _hostSurfaces;
        const Surface** _devSurfaces;
        const size_t _nsurf;
    };
}

#endif
