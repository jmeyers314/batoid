#include "quadric2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    Quadric2::Quadric2(double R, double conic) : _R(R), _conic(conic),
        _Rsq(R*R), _Rinvsq(1./R/R),
        _cp1(conic+1), _cp1inv(1./_cp1),
        _Rcp1(R/_cp1), _RRcp1cp1(R*R/_cp1/_cp1),
        _cp1RR(_cp1/R/R) {}

    #pragma omp declare target
    double Quadric2::_sag(double x, double y) const {
        double r2 = x*x + y*y;
        if (_R != 0)
            return r2/(_R*(1.+std::sqrt(1.-r2*_cp1RR)));
        return 0.0;
    }

    void Quadric2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
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

    bool Quadric2::_timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        double z0term = z-_Rcp1;
        double vrr0 = vx*x + vy*y;

        double a = vz*vz + (vx*vx+vy*vy)*_cp1inv;
        double b = 2*(vz*z0term + vrr0*_cp1inv);
        double c = z0term*z0term - _RRcp1cp1 + (x*x + y*y)*_cp1inv;

        double discriminant = b*b - 4*a*c;

        double dt1;
        if (b > 0) {
            dt1 = (-b - sqrt(discriminant)) / (2*a);
        } else {
            dt1 = 2*c / (-b + sqrt(discriminant));
        }
        double dt2 = c / (a*dt1);

        if (dt1 > 0) {
            dt = dt1;
            return true;
        }
        if (dt2 > 0) {
            dt = dt2;
            return true;
        }
        return false;
    }

    double Quadric2::_dzdr(double r) const {
        if (_R != 0.0)
            return r/(_R*std::sqrt(1-r*r*_cp1RR));
        return 0.0;
    }
    #pragma omp end declare target
}
