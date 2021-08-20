#ifndef batoid_tilted_h
#define batoid_tilted_h

#include "surface.h"

namespace batoid {

    class Tilted : public Surface {
    public:
        Tilted(double tanx, double tany);
        ~Tilted();

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
        const double _tanx, _tany;
    };

}
#endif
