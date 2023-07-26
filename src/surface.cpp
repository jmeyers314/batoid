#include "surface.h"
#include <iostream>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Surface::Surface() :
        _devPtr(nullptr)
    {}

    Surface::~Surface() {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                freeDevPtr();
            }
        #endif
    }

    bool Surface::timeToIntersect(
        const double x, const double y, const double z,
        const double vx, const double vy, const double vz,
        double& dt,  // Used as initial guess on input!
        int niter
    ) const {
        // The better the initial estimate of dt, the better this will perform
        double rPx = x+vx*dt;
        double rPy = y+vy*dt;
        double rPz = z+vz*dt;

        double sz = sag(rPx, rPy);

        // std::cout << "Surface::timeToIntersect  niter = " << niter << "\n";

        for (int iter=0; iter<niter; iter++) {
            // repeatedly intersect plane tangent to surface at (rPx, rPy, sz) with ray
            double nx, ny, nz;
            normal(rPx, rPy, nx, ny, nz);
            dt = (rPx-x)*nx + (rPy-y)*ny + (sz-z)*nz;
            dt /= (nx*vx + ny*vy + nz*vz);
            rPx = x+vx*dt;
            rPy = y+vy*dt;
            rPz = z+vz*dt;
            sz = sag(rPx, rPy);
        }
        return (std::abs(sz-rPz) < 1e-12);
    }

    void Surface::grad(
        double x, double y,
        double& dzdx, double& dzdy
    ) const {
        double nx, ny, nz;
        normal(x, y, nx, ny, nz);
        if (std::isnan(nx)) {
            dzdx = NAN;
            dzdy = NAN;
            return;
        }
        dzdx = -nx/nz;
        dzdy = -ny/nz;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    #if defined(BATOID_GPU)
    void Surface::freeDevPtr() const {
        if(_devPtr) {
            Surface* ptr = _devPtr;
            _devPtr = nullptr;
            #pragma omp target is_device_ptr(ptr)
            {
                delete ptr;
            }
        }
    }
    #endif

}
