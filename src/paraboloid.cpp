#include "paraboloid.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Paraboloid::Paraboloid(double R) :
        _R(R), _Rinv(1./R), _2Rinv(1./2/R)
    {}

    Paraboloid::~Paraboloid() {}

    double Paraboloid::sag(double x, double y) const {
        if (_R != 0) {
            double r2 = x*x + y*y;
            return r2*_2Rinv;
        }
        return 0.0;
    }

    void Paraboloid::normal(
        double x, double y,
        double& nx, double& ny, double& nz
    ) const {
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

    bool Paraboloid::timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        double a = (vx*vx + vy*vy)*_2Rinv;
        double b = (x*vx + y*vy)*_Rinv - vz;
        double c = (x*x + y*y)*_2Rinv - z;
        double dt1;

        if (a == 0) {
            // Ray is heading straight at optic axis.
            dt = -c/b;
            return true;
        }

        double discriminant = b*b - 4*a*c;

        if (discriminant < 0)
            return false;

        if (b > 0) {
            dt1 = (-b - sqrt(discriminant)) / (2*a);
        } else {
            dt1 = 2*c / (-b + sqrt(discriminant));
        }
        double dt2 = c / (a*dt1);

        // If vz down and paraboloid concave up, then pick greatest time;
        // I.e., intersection closest to the origin.
        // All other cases follow from symmetry
        dt = (vz*_R < 0) ? std::max(dt1, dt2) : std::min(dt1, dt2);
        return true;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Surface* Paraboloid::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Surface* ptr;
                #pragma omp target map(from:ptr)
                {
                    ptr = new Paraboloid(_R);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
