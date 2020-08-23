#ifndef batoid_surface_h
#define batoid_surface_h

#include "rayVector.h"
#include "coordSys.h"

namespace batoid {
    class Surface {
    public:
        virtual ~Surface() {}

        virtual Surface* getDevPtr() const = 0;

        virtual double sag(double x, double y) const = 0;
        virtual void normal(double x, double y, double& nx, double& ny, double& nz) const = 0;
        virtual bool timeToIntersect(double x, double y, double z, double vx, double vy, double vz, double& dt) const;

        void intersectInPlace(RayVector& rv, const CoordSys* cs=nullptr) const;
        // virtual void reflectInPlace(RayVector2&, const CoordSys* cs=nullptr) const = 0;
        // virtual void refractInPlace(RayVector2&, const Medium2&, const Medium2&, const CoordSys* cs=nullptr) const = 0;

    protected:
        Surface() :
            _devPtr(nullptr)
        {}
        mutable Surface* _devPtr;
    };
}
#endif
