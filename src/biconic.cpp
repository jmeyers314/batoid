#include "biconic.h"
#include "sphere.h"
#include <cmath>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Biconic::Biconic(double Rx, double Ry, double kx, double ky) :
        _Rx(Rx), _Ry(Ry),
        _kx(kx), _ky(ky),
        _cx(1.0 / Rx), _cy(1.0 / Ry) {}

    Biconic::~Biconic() {}

    double Biconic::sag(double x, double y) const {
        double x_term = _cx * _cx * (1 + _kx) * x * x;
        double y_term = _cy * _cy * (1 + _ky) * y * y;
        double sqrt_term = std::sqrt(1 - x_term - y_term);
        double denom = 1 + sqrt_term;

        if (denom <= 0) return 0.0;

        return (_cx * x * x + _cy * y * y) / denom;
    }

    void Biconic::normal(double x, double y, double& nx, double& ny, double& nz) const {
        double x_term = _cx * _cx * (1 + _kx) * x * x;
        double y_term = _cy * _cy * (1 + _ky) * y * y;
        double sqrt_term = std::sqrt(1.0 - x_term - y_term);
        double denom = sqrt_term + 1;

        if (std::abs(denom) < 1e-12) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
            return;
        }

        double dzdx = (2 * _cx * x) / denom +
                      (_cx * _cx * (1 + _kx) * x * (_cx * x * x + _cy * y * y)) /
                      (sqrt_term * denom * denom);

        double dzdy = (2 * _cy * y) / denom +
                      (_cy * _cy * (1 + _ky) * y * (_cy * y * y + _cx * x * x)) /
                      (sqrt_term * denom * denom);

        double dzdr_sq = dzdx * dzdx + dzdy * dzdy;
        nz = 1.0 / std::sqrt(1.0 + dzdr_sq);
        nx = -dzdx * nz;
        ny = -dzdy * nz;
    }

    bool Biconic::timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt, int niter
    ) const {
        // Use Sphere as a good initial guess
        double R_approx = (_Rx + _Ry) / 2.0;  // Approximate radius of curvature
        Sphere sphere(R_approx);
    
        if (!sphere.timeToIntersect(x, y, z, vx, vy, vz, dt, niter))
            return false;
    
        return Surface::timeToIntersect(x, y, z, vx, vy, vz, dt, niter);
    }
    

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Surface* Biconic::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Surface* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new Biconic(_Rx, _Ry, _kx, _ky);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
