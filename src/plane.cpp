#include "plane.h"

namespace batoid {

    #pragma omp declare target

    Plane::Plane() {}

    Plane::~Plane() {}

    double Plane::sag(double x, double y) const {
        return 0.0;
    }

    void Plane::normal(double x, double y, double& nx, double& ny, double& nz) const {
        nx = 0.0;
        ny = 0.0;
        nz = 1.0;
    }

    bool Plane::timeToIntersect(
        double x, double y, double z, double vx, double vy, double vz, double& dt
    ) const {
        if (vz == 0)
            return false;
        dt = -z/vz;
        return true;
    }

    #pragma omp end declare target


    Surface* Plane::getDevPtr() const {
        if (!_devPtr) {
            Surface* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new Plane();
            }
            _devPtr = ptr;
        }
        return _devPtr;
    }

}
