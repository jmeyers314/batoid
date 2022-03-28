#ifndef batoid_paraboloid_h
#define batoid_paraboloid_h

#include "surface.h"

namespace batoid {

    ////////////////
    // Paraboloid //
    ////////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Paraboloid : public Surface {
    public:
        Paraboloid(double R);
        ~Paraboloid();

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

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    //////////////////////
    // ParaboloidHandle //
    //////////////////////

    class ParaboloidHandle : public SurfaceHandle {
    public:
        ParaboloidHandle(double R);
        virtual ~ParaboloidHandle();
    };

}
#endif
