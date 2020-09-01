#ifndef batoid_bicubic_h
#define batoid_bicubic_h

#include "surface.h"

namespace batoid {

    class Bicubic : public Surface {
    public:
        Bicubic(
            double x0, double y0, double dx, double dy,
            const double* z, const double* dzdx, const double* dzdy, const double*d2zdxdy,
            size_t nx, size_t ny
        );
        ~Bicubic();

        virtual Surface* getDevPtr() const override;

        virtual double sag(double, double) const override;
        virtual void normal(double x, double y, double& nx, double& ny, double& nz) const override;
        // virtual bool timeToIntersect(double x, double y, double z, double vx, double vy, double vz, double& dt) const override;

    private:
        const double _x0, _y0;
        const double _dx, _dy;
        const double* _z;
        const double* _dzdx;
        const double* _dzdy;
        const double* _d2zdxdy;
        const size_t _nx, _ny;
    };

}
#endif
