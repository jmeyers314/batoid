#ifndef batoid_PolynomialSurface_h
#define batoid_PolynomialSurface_h

#include "surface.h"

namespace batoid {

    class PolynomialSurface : public Surface {
    public:
        PolynomialSurface(
            const double* coefs, const double* coefs_gradx, const double* coefs_grady,
            size_t nx, size_t ny
        );
        ~PolynomialSurface();

        virtual const Surface* getDevPtr() const override;

        virtual double sag(double, double) const override;
        virtual void normal(
            double x, double y,
            double& nx, double& ny, double& nz
        ) const override;

    private:
        const double* _coefs;
        const double* _coefs_gradx;
        const double* _coefs_grady;
        const size_t _nx, _ny;
    };

    double horner2d(double x, double y, const double* coefs, size_t nx, size_t ny);
}

#endif // batoid_PolynomialSurface_h
