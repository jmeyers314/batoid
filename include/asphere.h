#ifndef batoid_asphere_h
#define batoid_asphere_h

#include "surface.h"
#include "quadric.h"

namespace batoid {

    /////////////
    // Asphere //
    /////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    double* _computeAsphereDzDrCoefs(const double* coefs, const size_t size);

    class Asphere : public Quadric {
    public:
        Asphere(double R, double conic, const double* coefs, size_t size);
        Asphere(double R, double conic, const double* coefs, const double* dzdrcoefs, size_t size);
        ~Asphere();

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
        const double* _coefs;
        const double* _dzdrcoefs;
        const size_t _size;
        const bool _owns_dzdrcoefs;

        double _dzdr(double r) const;
    };

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////////
    // AsphereHandle //
    ///////////////////

    class AsphereHandle : public SurfaceHandle {
    public:
        AsphereHandle(double R, double conic, const double* coefs, size_t size);
        virtual ~AsphereHandle();
    private:
        const double* _coefs;
        const double* _dzdrcoefs;
        const size_t _size;
    };

}
#endif
