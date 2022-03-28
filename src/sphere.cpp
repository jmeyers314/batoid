#include "sphere.h"
#include <omp.h>

namespace batoid {

    ////////////
    // Sphere //
    ////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Sphere::Sphere(double R) :
        _R(R), _Rsq(R*R), _Rinv(1./R), _Rinvsq(1./R/R)
    {}

    Sphere::~Sphere() {}

    double Sphere::sag(double x, double y) const {
        if (_R != 0)
            return _R*(1-std::sqrt(1-(x*x + y*y)*_Rinvsq));
        return 0.0;
    }

    void Sphere::normal(double x, double y, double& nx, double& ny, double& nz) const {
        double rsqr = x*x + y*y;
        if (_R == 0.0 || rsqr == 0.0) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        } else {
            double r = sqrt(rsqr);
            double dzdr = _dzdr(r);
            nz = 1/sqrt(1+dzdr*dzdr);
            nx = -x/r*dzdr*nz;
            ny = -y/r*dzdr*nz;
        }
    }

    bool Sphere::timeToIntersect(
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

        if (discriminant < 0)
            return false;

        double dt1;
        if (b > 0) {
            dt1 = (-b - sqrt(discriminant)) / (2*a);
        } else {
            dt1 = 2*c / (-b + sqrt(discriminant));
        }
        double dt2 = c / (a*dt1);

        // If vz < 0, R > 0, want greatest time.
        dt = (vz*_R < 0) ? std::max(dt1, dt2) : std::min(dt1, dt2);
        return true;
    }

    double Sphere::_dzdr(double r) const {
        double rat = r*_Rinv;
        return rat/sqrt(1-rat*rat);
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    //////////////////
    // SphereHandle //
    //////////////////

    SphereHandle::SphereHandle(double R) : SurfaceHandle() {
        _hostPtr = new Sphere(R);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(Sphere), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) Sphere(R);
            }
        #endif
    }

    SphereHandle::~SphereHandle() {
        #if defined(BATOID_GPU)
            // We know following is a noop, but compiler might not.
            // This is what it'd look like though if we wanted to destruct on the device.

            // auto devPtr = static_cast<Sphere *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~Sphere();
            // }

            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }

}
