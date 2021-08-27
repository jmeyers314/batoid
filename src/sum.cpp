#include "sum.h"


namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Sum::Sum(const Surface** surfaces, size_t nsurf) :
        Surface(), _surfaces(surfaces), _nsurf(nsurf)
    {}

    Sum::~Sum() {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                Surface* ptr = _devPtr;
                #pragma omp target is_device_ptr(ptr)
                {
                    delete ptr;
                }
                const Surface** surfaces = _surfaces;
                #pragma omp target exit data map(release:surfaces[:_nsurf])
            }
        #endif
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


    const Surface* Sum::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            const Surface** surfaces = new const Surface*[_nsurf];
            for (int i=0; i<_nsurf; i++) {
                surfaces[i] = _surfaces[i]->getDevPtr();
            }
            Surface* ptr;
            #pragma omp target enter data map(to:surfaces[:_nsurf])
            #pragma omp target map(from:ptr)
            {
                ptr = new Sum(surfaces, _nsurf);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }

}
