#include "asphere.h"
#include "quadric.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    double* Asphere::_computeDzDrCoefs(const double* coefs, const size_t size) {
        double* result = new double[size];
        for(int i=4, j=0; j<size; j++, i += 2) {
            result[j] = coefs[j]*i;
        }
        return result;
    }

    Asphere::Asphere(double R, double conic, const double* coefs, size_t size) :
        Quadric(R, conic),
        _coefs(coefs),
        _dzdrcoefs(_computeDzDrCoefs(coefs, size)),
        _size(size)
    {}

    Asphere::~Asphere()
    {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                Surface* ptr = _devPtr;
                #pragma omp target is_device_ptr(ptr)
                {
                    delete ptr;
                }
                const size_t size = _size;
                const double* coefs = _coefs;
                #pragma omp target exit data map(release:coefs[:size])
            }
        #endif
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

    const Surface* Asphere::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Surface* ptr;
                // Allocate coef array on device
                const double* coefs = _coefs;
                #pragma omp target enter data map(to:coefs[:_size])
                #pragma omp target map(from:ptr)
                {
                    ptr = new Asphere(_R, _conic, coefs, _size);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
