#ifndef batoid_asphere_h
#define batoid_asphere_h

#include "surface.h"
#include "quadric.h"

namespace batoid {

    class Asphere : public Quadric {
    public:
        Asphere(double R, double conic, const double* coefs, size_t size);
        ~Asphere();

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
        const double* _coefs;
        const double* _dzdrcoefs;
        const size_t _size;

        double _dzdr(double r) const;
        static double* _computeDzDrCoefs(const double* coefs, const size_t size);
    };

}
#endif
