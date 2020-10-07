#ifndef batoid_sum_h
#define batoid_sum_h

#include "surface.h"

namespace batoid {

    class Sum : public Surface {
    public:
        Sum(const Surface** surfaces, size_t nsurf);
        ~Sum();

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
        const Surface** _surfaces;
        size_t _nsurf;
    };

}

#endif
