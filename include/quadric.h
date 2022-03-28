#ifndef batoid_quadric_h
#define batoid_quadric_h

#include "surface.h"

namespace batoid {

    /////////////
    // Quadric //
    /////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    class Quadric : public Surface {
    public:
        Quadric(double R, double conic);
        ~Quadric();

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

    protected:
        const double _R;  // Radius of curvature
        const double _conic;  // Conic constant

        double _dzdr(double r) const;

    private:
        const double _Rsq;  // R*R
        const double _Rinvsq;  // 1/R/R
        const double _cp1; // 1 + conic
        const double _cp1inv; // 1/(1 + conic)
        const double _Rcp1; // R/(1+conic)
        const double _RRcp1cp1; // R*R/(1+conic)/(1+conic)
        const double _cp1RR; // (1+conic)/R/R
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    ///////////////////
    // QuadricHandle //
    ///////////////////

    class QuadricHandle : public SurfaceHandle {
    public:
        QuadricHandle(double R, double conic);
        virtual ~QuadricHandle();
    };

}
#endif
