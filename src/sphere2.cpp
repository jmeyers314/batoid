#include "sphere2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    Sphere2::Sphere2(double R) : _R(R), _Rsq(R*R), _Rinv(1./R), _Rinvsq(1./R/R) {}

    #pragma omp declare target
    double Sphere2::_sag(double x, double y) const {
        if (_R != 0)
            return _R*(1-sqrt(1-(x*x + y*y)*_Rinvsq));
        return 0.0;
    }

    void Sphere2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
        double rsqr = x*x + y*y;
        if (_R == 0.0 || rsqr == 0.0) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        } else {
            double r = sqrt(rsqr);
            double dzdr = _dzdr(r);
            double norm = 1/sqrt(1+dzdr*dzdr);
            nx = -x/r*dzdr*norm;
            ny = -y/r*dzdr*norm;
            nz = norm;
        }
    }

    bool Sphere2::_timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        double vr2 = vx*vx + vy*vy;
        double vz2 = vz*vz;
        double vrr0 = vx*x + vy*y;
        double r02 = x*x + y*y;
        double z0term = z-_R;

        double a = vz2 + vr2;
        double b = 2*vz*z0term + 2*vrr0;
        double c = z0term*z0term - _Rsq + r02;

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

    double Sphere2::_dzdr(double r) const {
        double rat = r*_Rinv;
        return rat/sqrt(1-rat*rat);
    }
    #pragma omp end declare target
}
