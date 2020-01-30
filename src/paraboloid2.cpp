#include "paraboloid2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    Paraboloid2::Paraboloid2(double R) : _R(R), _Rinv(1./R), _2Rinv(1./2/R) {}

    #pragma omp declare target
    double Paraboloid2::_sag(double x, double y) const {
        if (_R != 0) {
            double r2 = x*x + y*y;
            return r2*_2Rinv;
        }
        return 0.0;
    }

    void Paraboloid2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
        if (_R == 0) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        } else {
            nz = 1/std::sqrt(1+(x*x+y*y)*_Rinv*_Rinv);
            nx = -x*_Rinv*nz;
            ny = -y*_Rinv*nz;
        }
    }

    bool Paraboloid2::_timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        double a = (vx*vx + vy*vy)*_2Rinv;
        double b = (x*vx + y*vy)*_Rinv - vz;
        double c = (x*x + y*y)*_2Rinv - z;

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
    #pragma omp end declare target
}
