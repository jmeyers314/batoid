#include "tilted.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Tilted::Tilted(double tanx, double tany) :
        _tanx(tanx), _tany(tany) {}

    Tilted::~Tilted() {}

    double Tilted::sag(double x, double y) const {
        return x*_tanx + y*_tany;
    }

    void Tilted::normal(
        double x, double y,
        double& nx, double& ny, double& nz
    ) const {
        nx = -_tanx;
        ny = -_tany;
        nz = sqrt(1 - _tanx*_tanx - _tany*_tany);
    }

    bool Tilted::timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        double num = z - x*_tanx - y*_tany;
        double den = vx*_tanx + vy*_tany - vz;
        if (den == 0) return false;
        dt = num/den;
        return true;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Surface* Tilted::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Surface* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new Tilted(_tanx, _tany);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
