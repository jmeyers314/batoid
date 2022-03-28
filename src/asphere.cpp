#include "asphere.h"
#include "quadric.h"
#include <omp.h>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    double* _computeAsphereDzDrCoefs(const double* coefs, const size_t size) {
        double* result = new double[size];
        for(int i=4, j=0; j<size; j++, i += 2) {
            result[j] = coefs[j]*i;
        }
        return result;
    }


    /////////////
    // Asphere //
    /////////////

    Asphere::Asphere(double R, double conic, const double* coefs, size_t size) :
        Quadric(R, conic),
        _coefs(coefs),
        _dzdrcoefs(_computeAsphereDzDrCoefs(coefs, size)),
        _size(size),
        _owns_dzdrcoefs(true)
    {}

    Asphere::Asphere(
        double R, double conic, const double* coefs, const double* dzdrcoefs, size_t size
    ) :
        Quadric(R, conic),
        _coefs(coefs),
        _dzdrcoefs(dzdrcoefs),
        _size(size),
        _owns_dzdrcoefs(false)
    {}

    Asphere::~Asphere()
    {
        if (_owns_dzdrcoefs)
            delete[] _dzdrcoefs;
    }

    double Asphere::sag(double x, double y) const {
        double r2 = x*x + y*y;
        double rr = r2;
        double result = Quadric::sag(x, y);
        for (int i=0; i<_size; i++) {
            rr *= r2;
            result += _coefs[i]*rr;
        }
        return result;
    }

    void Asphere::normal(
        double x, double y,
        double& nx, double& ny, double& nz
    ) const {
        double r = std::sqrt(x*x + y*y);
        if (r == 0.0) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        } else {
            double dzdr = _dzdr(r);
            nz = 1/sqrt(1+dzdr*dzdr);
            nx = -x/r*dzdr*nz;
            ny = -y/r*dzdr*nz;
        }
    }

    double Asphere::_dzdr(double r) const {
        double result = Quadric::_dzdr(r);
        double rr = r*r;
        double rrr = rr*r;
        for (int i=0; i<_size; i++) {
            result += _dzdrcoefs[i]*rrr;
            rrr *= rr;
        }
        return result;
    }

    bool Asphere::timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        // Solve the quadric problem analytically to get a good starting point.
        if (!Quadric::timeToIntersect(x, y, z, vx, vy, vz, dt))
            return false;
        return Surface::timeToIntersect(x, y, z, vx, vy, vz, dt);
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    ///////////////////
    // AsphereHandle //
    ///////////////////

    AsphereHandle::AsphereHandle(double R, double conic, const double* coefs, size_t size) :
        SurfaceHandle(),
        _coefs(coefs),
        _dzdrcoefs(_computeAsphereDzDrCoefs(coefs, size)),
        _size(size)
    {
        _hostPtr = new Asphere(R, conic, _coefs, _dzdrcoefs, _size);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(Asphere), omp_get_default_device());
            const double* cfs = _coefs;
            const double* dzdrcfs = _dzdrcoefs;
            #pragma omp target enter data map(to:cfs[:_size], dzdrcfs[:_size])
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) Asphere(R, conic, cfs, dzdrcfs, _size);
            }
        #endif
    }

    AsphereHandle::~AsphereHandle() {
        #if defined(BATOID_GPU)
            // We know following is noop, but compiler might not...

            // auto devPtr = static_cast<Asphere *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~Asphere();
            // }

            #pragma omp target exit data map(release:_coefs[:_size], _dzdrcoefs[:_size])
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete[] _dzdrcoefs;
        delete _hostPtr;
    }
}
