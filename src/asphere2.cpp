#include "asphere2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    Asphere2::Asphere2(double R, double conic, const std::array<double, 10>& coefs) :
        _coefs(coefs), _dzdrcoefs(_computeDzDrCoefs(coefs)), _q(Quadric2(R, conic)) {}

    std::array<double,10> Asphere2::_computeDzDrCoefs(const std::array<double,10>& coefs) {
        std::array<double,10> result;
        for (int i=0, j=4; i<10; i++, j+=2) {
            result[i] = coefs[i]*j;
        }
        return result;
    }

    #pragma omp declare target
    double Asphere2::_sag(double x, double y) const {
        double r2 = x*x + y*y;
        double rr = r2;
        double result = _q._sag(x, y);
        for (const auto& c : _coefs) {
            rr *= r2;
            result += c*rr;
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
        for (const auto& c : _dzdrcoefs) {
            result += c*rrr;
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
