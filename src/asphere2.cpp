#include "asphere2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    Asphere2::Asphere2(double R, double conic, double* coefs, size_t size) :
        _coefs(coefs, size), _dzdrcoefs(size), _size(size), _q(Quadric2(R, conic)) {
            _computeDzDrCoefs();
            _coefs.syncToDevice();
            _dzdrcoefs.syncToDevice();
        }

    void Asphere2::_computeDzDrCoefs() {
        // Do computation on host
        _coefs.syncToHost();
        _dzdrcoefs.owner = DVOwnerType::host;

        for (int i=0, j=4; i<_size; i++, j+=2) {
            _dzdrcoefs.hostData[i] = _coefs.hostData[i]*j;
        }
    }

    #pragma omp declare target
    double Asphere2::_sag(double x, double y) const {
        double r2 = x*x + y*y;
        double rr = r2;
        double result = _q._sag(x, y);
        for (int i=0; i<_size; i++) {
            rr *= r2;
            result += _coefs.deviceData[i]*rr;
        }
        return result;
    }

    void Asphere2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
        double r = std::sqrt(x*x + y*y);
        if (r == 0.0) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        } else {
            double dzdr = _dzdr(r);
            double norm = 1/sqrt(1+dzdr*dzdr);
            nx = -x/r*dzdr*norm;
            ny = -y/r*dzdr*norm;
            nz = norm;
        }
    }

    double Asphere2::_dzdr(double r) const {
        double result = _q._dzdr(r);
        double rr = r*r;
        double rrr = rr*r;
        for (int i=0; i<_size; i++) {
            result += _dzdrcoefs.deviceData[i]*rrr;
            rrr *= rr;
        }
        return result;
    }

    bool Asphere2::_timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        // Solve the quadric problem analytically to get a good starting point.
        if (!_q._timeToIntersect(x, y, z, vx, vy, vz, dt))
            return false;
        // follow with Newton-Raphson to refine initial estimate.
        double rPx = x+vx*dt;
        double rPy = y+vy*dt;
        double rPz = z+vz*dt;

        double sz = _sag(rPx, rPy);
        for (int iter=0; iter<10; iter++) {
            // repeatedly intersect plane tangent to surface at (rPx, rPy, sz) with ray
            double nx, ny, nz;
            _normal(rPx, rPy, nx, ny, nz);
            dt = (rPx-x)*nx + (rPy-y)*ny + (sz-z)*nz;
            dt /= (nx*vx + ny*vy + nz*vz);
            rPx = x+vx*dt;
            rPy = y+vy*dt;
            rPz = z+vz*dt;
            sz = _sag(rPx, rPy);
        }
        return (abs(sz-rPz) < 1e-14);
    }
    #pragma omp end declare target
}
