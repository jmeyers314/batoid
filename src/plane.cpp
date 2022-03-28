#include "plane.h"
#include <omp.h>

namespace batoid {

    ///////////
    // Plane //
    ///////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

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

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    /////////////////
    // PlaneHandle //
    /////////////////

    PlaneHandle::PlaneHandle() : SurfaceHandle() {
        _hostPtr = new Plane();
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(Plane), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) Plane();
            }
        #endif
    }

    PlaneHandle::~PlaneHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<Plane *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~Plane();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }
}
