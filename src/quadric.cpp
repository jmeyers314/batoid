#include "quadric.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Quadric::Quadric(double R, double conic) :
        _R(R), _conic(conic),
        _Rsq(R*R), _Rinvsq(1./R/R),
        _cp1(conic+1), _cp1inv(1./_cp1),
        _Rcp1(R/_cp1), _RRcp1cp1(R*R/_cp1/_cp1),
        _cp1RR(_cp1/R/R) {}

    Quadric::~Quadric() {}

    double Quadric::sag(double x, double y) const {
        double r2 = x*x + y*y;
        if (_R != 0)
            return r2/(_R*(1.+std::sqrt(1.-r2*_cp1RR)));
        return 0.0;
        // Following almost works, except leads to divide by 0 when _conic=-1
        // return R/(1+_conic)*(1-std::sqrt(1-(1+_conic)*r2/R/R));
    }

    void Quadric::normal(double x, double y, double& nx, double& ny, double& nz) const {
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

    bool Quadric::timeToIntersect(
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

        if (discriminant < 0)
            return false;

        double dt1;
        if (b > 0) {
            dt1 = (-b - sqrt(discriminant)) / (2*a);
        } else {
            dt1 = 2*c / (-b + sqrt(discriminant));
        }
        double dt2 = c / (a*dt1);

        // New strategy, just always pick smaller abs(z).
        double z1 = z + vz*dt1;
        double z2 = z + vz*dt2;
        dt = (std::abs(z1) < std::abs(z2)) ? dt1 : dt2;
        return true;
    }

    double Quadric::_dzdr(double r) const {
        if (_R != 0.0)
            return r/(_R*std::sqrt(1-r*r*_cp1RR));
        return 0.0;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    const Surface* Quadric::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Surface* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new Quadric(_R, _conic);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }


}
