#ifndef batoid_biconic_h
#define batoid_biconic_h

#include "surface.h"
#include "sphere.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Biconic : public Surface {
    public:
        Biconic(double Rx, double Ry, double kx, double ky);
        ~Biconic();

        virtual const Surface* getDevPtr() const override;

        virtual double sag(double x, double y) const override;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const override;
        virtual bool timeToIntersect(
            double x, double y, double z,
            double vx, double vy, double vz,
            double& dt, int niter
        ) const override;

    private:
        const double _Rx, _Ry;  // Radii of curvature in x and y
        const double _kx, _ky;  // Conic constants in x and y
        const double _cx, _cy;  // Curvature (1/R)

        mutable const Surface* _devPtr = nullptr;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

}
#endif