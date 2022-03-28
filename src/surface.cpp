#include "surface.h"

namespace batoid {

    /////////////
    // Surface //
    /////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Surface::Surface() {}

    Surface::~Surface() {}

    bool Surface::timeToIntersect(
        const double x, const double y, const double z,
        const double vx, const double vy, const double vz,
        double& dt  // Used as initial guess on input!
    ) const {
        // The better the initial estimate of dt, the better this will perform
        double rPx = x+vx*dt;
        double rPy = y+vy*dt;
        double rPz = z+vz*dt;

        double sz = sag(rPx, rPy);
        // Always do exactly 5 iterations.  GPUifies better this way.
        // Unit tests pass (as of 20/10/13) with just 3 iterations.
        // Algorithm is 2D Newton-Raphson iterations
        for (int iter=0; iter<5; iter++) {
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


    ///////////////////
    // SurfaceHandle //
    ///////////////////

    SurfaceHandle::SurfaceHandle() :
        _hostPtr(nullptr),
        _devicePtr(nullptr)
    {}

    SurfaceHandle::~SurfaceHandle() {}

    const Surface* SurfaceHandle::getPtr() const {
        #if defined(BATOID_GPU)
            return _devicePtr;
        #else
            return _hostPtr;
        #endif
    }

    const Surface* SurfaceHandle::getHostPtr() const {
        return _hostPtr;
    }



}
