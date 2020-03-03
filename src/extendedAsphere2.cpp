#include "extendedAsphere2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    ExtendedAsphere2::ExtendedAsphere2(
        double R, double conic, double* coefs, size_t ncoefs,
        double x0, double y0, double dx, double dy,
        double* z, double* dzdx, double* dzdy, double* d2zdxdy,
        size_t gridsize
    ) :
        _coefs(coefs, ncoefs), _dzdrcoefs(ncoefs), _ncoefs(ncoefs), _q(Quadric2(R, conic)),
        _x0(x0), _y0(y0), _dx(dx), _dy(dy),
        _z(z, gridsize*gridsize), _dzdx(dzdx, gridsize*gridsize), _dzdy(dzdy, gridsize*gridsize),
        _d2zdxdy(d2zdxdy, gridsize*gridsize),
        _gridsize(gridsize)
        {
            _computeDzDrCoefs();
            _coefs.syncToDevice();
            _dzdrcoefs.syncToDevice();
            _z.syncToDevice();
            _dzdx.syncToDevice();
            _dzdy.syncToDevice();
            _d2zdxdy.syncToDevice();
        }

    void ExtendedAsphere2::_computeDzDrCoefs() {
        // Do computation on host
        _coefs.syncToHost();
        _dzdrcoefs.owner = DVOwnerType::host;

        for (int i=0, j=4; i<_ncoefs; i++, j+=2) {
            _dzdrcoefs.hostData[i] = _coefs.hostData[i]*j;
        }
    }

    #pragma omp declare target
    double ExtendedAsphere2::oneDSpline(double x, double val0, double val1, double der0, double der1) const {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        double d = val0;

        return d + x*(c + x*(b + x*a));
    }

    double ExtendedAsphere2::oneDGrad(double x, double val0, double val1, double der0, double der1) const {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        return c + x*(2*b + x*3*a);
    }

    double ExtendedAsphere2::_sag(double x, double y) const {
        // Asphere part
        double r2 = x*x + y*y;
        double rr = r2;
        double result = _q._sag(x, y);
        for (int i=0; i<_ncoefs; i++) {
            rr *= r2;
            result += _coefs.deviceData[i]*rr;
        }

        // Bicubic part
        int ix = int(std::floor((x-_x0)/_dx));
        int iy = int(std::floor((y-_y0)/_dy));
        double xgrid = _x0 + ix*_dx;
        double ygrid = _y0 + iy*_dy;
        double xfrac = (x - xgrid)/_dx;
        double yfrac = (y - ygrid)/_dy;
        double* zptr = _z.deviceData;
        double* dzdxptr = _dzdx.deviceData;
        double* dzdyptr = _dzdy.deviceData;
        double* d2zdxdyptr = _d2zdxdy.deviceData;

        double val0 = oneDSpline(
            xfrac,
               zptr[_gridsize*iy + ix],        zptr[_gridsize*iy + ix+1],
            dzdxptr[_gridsize*iy + ix]*_dx, dzdxptr[_gridsize*iy + ix+1]*_dx
        );
        double val1 = oneDSpline(
            xfrac,
               zptr[_gridsize*(iy+1) + ix],        zptr[_gridsize*(iy+1) + ix+1],
            dzdxptr[_gridsize*(iy+1) + ix]*_dx, dzdxptr[_gridsize*(iy+1) + ix+1]*_dx
        );
        double der0 = oneDSpline(
            xfrac,
               dzdyptr[_gridsize*iy + ix],        dzdyptr[_gridsize*iy + ix+1],
            d2zdxdyptr[_gridsize*iy + ix]*_dx, d2zdxdyptr[_gridsize*iy + ix+1]*_dx
        );
        double der1 = oneDSpline(
            xfrac,
               dzdyptr[_gridsize*(iy+1) + ix],        dzdyptr[_gridsize*(iy+1) + ix+1],
            d2zdxdyptr[_gridsize*(iy+1) + ix]*_dx, d2zdxdyptr[_gridsize*(iy+1) + ix+1]*_dx
        );
        result += oneDSpline(yfrac, val0, val1, der0*_dy, der1*_dy);
        return result;
    }

    void ExtendedAsphere2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
        // Asphere part
        double r = std::sqrt(x*x + y*y);
        double gradx, grady;
        if (r == 0.0) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
            return;
        }

        double dzdr = _dzdr(r);
        gradx = x/r*dzdr;
        grady = y/r*dzdr;

        // Bicubic part
        int ix = int(std::floor((x-_x0)/_dx));
        int iy = int(std::floor((y-_y0)/_dy));
        double xgrid = _x0 + ix*_dx;
        double ygrid = _y0 + iy*_dy;
        double xfrac = (x - xgrid)/_dx;
        double yfrac = (y - ygrid)/_dy;
        double* zptr = _z.deviceData;
        double* dzdxptr = _dzdx.deviceData;
        double* dzdyptr = _dzdy.deviceData;
        double* d2zdxdyptr = _d2zdxdy.deviceData;

        // x-gradient
        double val0 = oneDGrad(
            xfrac,
               zptr[_gridsize*iy + ix],        zptr[_gridsize*iy + ix+1],
            dzdxptr[_gridsize*iy + ix]*_dx, dzdxptr[_gridsize*iy + ix+1]*_dx
        );
        double val1 = oneDGrad(
            xfrac,
               zptr[_gridsize*(iy+1) + ix],        zptr[_gridsize*(iy+1) + ix+1],
            dzdxptr[_gridsize*(iy+1) + ix]*_dx, dzdxptr[_gridsize*(iy+1) + ix+1]*_dx
        );
        double der0 = oneDGrad(
            xfrac,
               dzdyptr[_gridsize*iy + ix],        dzdyptr[_gridsize*iy + ix+1],
            d2zdxdyptr[_gridsize*iy + ix]*_dx, d2zdxdyptr[_gridsize*iy + ix+1]*_dx
        );
        double der1 = oneDGrad(
            xfrac,
               dzdyptr[_gridsize*(iy+1) + ix],        dzdyptr[_gridsize*(iy+1) + ix+1],
            d2zdxdyptr[_gridsize*(iy+1) + ix]*_dx, d2zdxdyptr[_gridsize*(iy+1) + ix+1]*_dx
        );
        gradx += oneDSpline(yfrac, val0, val1, der0*_dy, der1*_dy)/_dx;

        // y-gradient
        val0 = oneDGrad(
            yfrac,
               zptr[_gridsize*iy + ix],        zptr[_gridsize*(iy+1) + ix],
            dzdyptr[_gridsize*iy + ix]*_dy, dzdyptr[_gridsize*(iy+1) + ix]*_dy
        );
        val1 = oneDGrad(
            yfrac,
               zptr[_gridsize*iy + ix+1],        zptr[_gridsize*(iy+1) + ix+1],
            dzdyptr[_gridsize*iy + ix+1]*_dy, dzdyptr[_gridsize*(iy+1) + ix+1]*_dy
        );
        der0 = oneDGrad(
            yfrac,
               dzdxptr[_gridsize*iy + ix],        dzdxptr[_gridsize*(iy+1) + ix],
            d2zdxdyptr[_gridsize*iy + ix]*_dy, d2zdxdyptr[_gridsize*(iy+1) + ix]*_dy
        );
        der1 = oneDGrad(
            yfrac,
               dzdxptr[_gridsize*iy + ix+1],        dzdxptr[_gridsize*(iy+1) + ix+1],
            d2zdxdyptr[_gridsize*iy + ix+1]*_dy, d2zdxdyptr[_gridsize*(iy+1) + ix+1]*_dy
        );
        grady += oneDSpline(xfrac, val0, val1, der0*_dx, der1*_dx)/_dy;

        // This works
        double norm = gradx*gradx;
        norm += grady*grady;
        norm += 1;
        norm = 1/std::sqrt(norm);

        nx = -gradx*norm;
        ny = -grady*norm;
        nz = norm;
    }

    double ExtendedAsphere2::_dzdr(double r) const {
        double result = _q._dzdr(r);
        double rr = r*r;
        double rrr = rr*r;
        for (int i=0; i<_ncoefs; i++) {
            result += _dzdrcoefs.deviceData[i]*rrr;
            rrr *= rr;
        }
        return result;
    }

    bool ExtendedAsphere2::_timeToIntersect(
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
