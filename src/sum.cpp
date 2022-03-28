#include "sum.h"
#include <omp.h>


namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Sum::Sum(const Surface** surfaces, size_t nsurf) :
        _surfaces(surfaces), _nsurf(nsurf)
    {}

    Sum::~Sum() {
    }

    double Sum::sag(double x, double y) const {
        double result = 0.0;
        for (int i=0; i<_nsurf; i++)
            result += _surfaces[i]->sag(x, y);
        return result;
    }

    void Sum::normal(
        double x, double y,
        double& nx, double& ny, double& nz
    ) const {
        nx = 0.0;
        ny = 0.0;
        for (int i=0; i<_nsurf; i++) {
            double tnx, tny, tnz;
            _surfaces[i]->normal(x, y, tnx, tny, tnz);
            nx += tnx/tnz;
            ny += tny/tnz;
        }
        nz = 1./std::sqrt(nx*nx + ny*ny + 1);
        nx *= nz;
        ny *= nz;
    }

    bool Sum::timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        // Use first surface as an initial guess
        if (!_surfaces[0]->timeToIntersect(x, y, z, vx, vy, vz, dt))
            return false;
        return Surface::timeToIntersect(x, y, z, vx, vy, vz, dt);
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////
    // SumHandle //
    ///////////////

    const Surface** SumHandle::_getSurfaces(
        const SurfaceHandle** handles, const size_t nsurf, bool host
    ) {
        auto out = new const Surface*[nsurf];
        for (size_t i=0; i<nsurf; i++) {
            out[i] = host ? handles[i]->getHostPtr() : handles[i]->getPtr();
        }
        return out;
    }

    SumHandle::SumHandle(const SurfaceHandle** handles, const size_t nsurf) :
        SurfaceHandle(),
        _hostSurfaces(_getSurfaces(handles, nsurf, true)),
        _devSurfaces(_getSurfaces(handles, nsurf, false)),
        _nsurf(nsurf)
    {
        _hostPtr = new Sum(_hostSurfaces, _nsurf);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(Sum), omp_get_default_device());
            const Surface** devS = _devSurfaces;
            size_t ns = _nsurf;
            #pragma omp target enter data map(to:devS[:_nsurf])
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) Sum(devS, ns);
            }
        #endif
    }

    SumHandle::~SumHandle() {
        #if defined(BATOID_GPU)
            // We know following is noop, but compiler might not...

            // auto devPtr = static_cast<Sum *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~Sum();
            // }

            #pragma omp target exit data map(release:_devSurfaces[:_nsurf])
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete[] _hostSurfaces;
        delete[] _devSurfaces;
        delete _hostPtr;
    }

}
