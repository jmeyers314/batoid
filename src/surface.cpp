#include "surface.h"

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Surface::Surface() :
        _devPtr(nullptr)
    {}

    Surface::~Surface() {}

    bool Surface::timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        // The better the initial estimate of dt, the better this will perform
        double rPx = x+vx*dt;
        double rPy = y+vy*dt;
        double rPz = z+vz*dt;

        double sz = sag(rPx, rPy);
        // Always do exactly 5 iterations.  GPUifies better this way.
        // Unit tests pass (as of 20/10/13) with just 3 iterations.
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

}
