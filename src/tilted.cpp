#include "tilted.h"
#include <omp.h>

namespace batoid {

    ////////////
    // Tilted //
    ////////////

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


    //////////////////
    // TiltedHandle //
    //////////////////

    TiltedHandle::TiltedHandle(double tanx, double tany) : SurfaceHandle() {
        _hostPtr = new Tilted(tanx, tany);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(Tilted), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) Tilted(tanx, tany);
            }
        #endif
    }

    TiltedHandle::~TiltedHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<Tilted *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~Tilted();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }
}
