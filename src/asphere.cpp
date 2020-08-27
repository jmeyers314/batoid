#include "asphere.h"
#include "quadric.h"

namespace batoid {

    #pragma omp declare target

    Asphere::Asphere(double R, double conic, const double* coefptr, size_t size) :
        Quadric(R, conic),
        _coefs(_copyCoefs(coefptr, size)),
        _dzdrcoefs(_computeDzDrCoefs(coefptr, size)),
        _size(size)
    {}

    Asphere::~Asphere()
    {
        delete[] _coefs;
        delete[] _dzdrcoefs;
    }

    double* Asphere::_copyCoefs(const double* coefs, const size_t size) {
        double* out = new double[size];
        for(int i=0; i<size; i++)
            out[i] = coefs[i];
        return out;
    }

    double* Asphere::_computeDzDrCoefs(const double* coefs, const size_t size) {
        double* result = new double[size];
        for(int i=4, j=0; j<size; j++, i += 2) {
            result[j] = coefs[j]*i;
        }
        return result;
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

    void Asphere::normal(double x, double y, double& nx, double& ny, double& nz) const {
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
        bool success = Surface::timeToIntersect(x, y, z, vx, vy, vz, dt);
        return (success && dt >= 0.0);
    }

    #pragma omp end declare target

    void Asphere::getCoefs(double* out) const {
        for(int i=0; i<_size; i++)
            out[i] = _coefs[i];
    }

    int Asphere::getSize() const {
        return _size;
    }

    Surface* Asphere::getDevPtr() const {
        if (!_devPtr) {
            Surface* ptr;
            // omp doesn't seem to find _R and _conic inherited from Quadric, so grab them
            // explicitly here.
            double R = getR();
            double conic = getConic();
            #pragma omp target map(from:ptr) map(to:_coefs[:_size])
            {
                ptr = new Asphere(R, conic, _coefs, _size);
            }
            _devPtr = ptr;
        }
        return _devPtr;
    }

}
