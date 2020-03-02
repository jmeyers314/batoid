#include "bicubic2.h"

namespace batoid {
    Bicubic2::Bicubic2(
        double x0, double y0, double dx, double dy,
        double* z, double* dzdx, double* dzdy, double*d2zdxdy,
        size_t size
    ) :
        _x0(x0), _y0(y0), _dx(dx), _dy(dy),
        _z(z, size*size), _dzdx(dzdx, size*size), _dzdy(dzdy, size*size),
        _d2zdxdy(d2zdxdy, size*size),
        _size(size)
    {
        _z.syncToDevice();
        _dzdx.syncToDevice();
        _dzdy.syncToDevice();
        _d2zdxdy.syncToDevice();
    }

    #pragma omp declare target
    double Bicubic2::oneDSpline(double x, double val0, double val1, double der0, double der1) const {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        double d = val0;

        return d + x*(c + x*(b + x*a));
    }

    double Bicubic2::oneDGrad(double x, double val0, double val1, double der0, double der1) const {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        return c + x*(2*b + x*3*a);
    }

    double Bicubic2::_sag(double x, double y) const {
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
               zptr[_size*iy + ix],        zptr[_size*iy + ix+1],
            dzdxptr[_size*iy + ix]*_dx, dzdxptr[_size*iy + ix+1]*_dx
        );
        double val1 = oneDSpline(
            xfrac,
               zptr[_size*(iy+1) + ix],        zptr[_size*(iy+1) + ix+1],
            dzdxptr[_size*(iy+1) + ix]*_dx, dzdxptr[_size*(iy+1) + ix+1]*_dx
        );
        double der0 = oneDSpline(
            xfrac,
               dzdyptr[_size*iy + ix],        dzdyptr[_size*iy + ix+1],
            d2zdxdyptr[_size*iy + ix]*_dx, d2zdxdyptr[_size*iy + ix+1]*_dx
        );
        double der1 = oneDSpline(
            xfrac,
               dzdyptr[_size*(iy+1) + ix],        dzdyptr[_size*(iy+1) + ix+1],
            d2zdxdyptr[_size*(iy+1) + ix]*_dx, d2zdxdyptr[_size*(iy+1) + ix+1]*_dx
        );
        return oneDSpline(yfrac, val0, val1, der0*_dy, der1*_dy);
    }

    void Bicubic2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
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
               zptr[_size*iy + ix],        zptr[_size*iy + ix+1],
            dzdxptr[_size*iy + ix]*_dx, dzdxptr[_size*iy + ix+1]*_dx
        );
        double val1 = oneDGrad(
            xfrac,
               zptr[_size*(iy+1) + ix],        zptr[_size*(iy+1) + ix+1],
            dzdxptr[_size*(iy+1) + ix]*_dx, dzdxptr[_size*(iy+1) + ix+1]*_dx
        );
        double der0 = oneDGrad(
            xfrac,
               dzdyptr[_size*iy + ix],        dzdyptr[_size*iy + ix+1],
            d2zdxdyptr[_size*iy + ix]*_dx, d2zdxdyptr[_size*iy + ix+1]*_dx
        );
        double der1 = oneDGrad(
            xfrac,
               dzdyptr[_size*(iy+1) + ix],        dzdyptr[_size*(iy+1) + ix+1],
            d2zdxdyptr[_size*(iy+1) + ix]*_dx, d2zdxdyptr[_size*(iy+1) + ix+1]*_dx
        );
        double gradx = oneDSpline(yfrac, val0, val1, der0*_dy, der1*_dy)/_dx;

        // y-gradient
        val0 = oneDGrad(
            yfrac,
               zptr[_size*iy + ix],        zptr[_size*(iy+1) + ix],
            dzdyptr[_size*iy + ix]*_dy, dzdyptr[_size*(iy+1) + ix]*_dy
        );
        val1 = oneDGrad(
            yfrac,
               zptr[_size*iy + ix+1],        zptr[_size*(iy+1) + ix+1],
            dzdyptr[_size*iy + ix+1]*_dy, dzdyptr[_size*(iy+1) + ix+1]*_dy
        );
        der0 = oneDGrad(
            yfrac,
               dzdxptr[_size*iy + ix],        dzdxptr[_size*(iy+1) + ix],
            d2zdxdyptr[_size*iy + ix]*_dy, d2zdxdyptr[_size*(iy+1) + ix]*_dy
        );
        der1 = oneDGrad(
            yfrac,
               dzdxptr[_size*iy + ix+1],        dzdxptr[_size*(iy+1) + ix+1],
            d2zdxdyptr[_size*iy + ix+1]*_dy, d2zdxdyptr[_size*(iy+1) + ix+1]*_dy
        );
        double grady = oneDSpline(xfrac, val0, val1, der0*_dx, der1*_dx)/_dy;

        // Following fails
        // double norm = 1/std::sqrt(gradx*gradx + grady+grady + 1.0);

        // This works
        double norm = gradx*gradx;
        norm += grady*grady;
        norm += 1;
        norm = 1/std::sqrt(norm);

        nx = -gradx*norm;
        ny = -grady*norm;
        nz = norm;
    }

    bool Bicubic2::_timeToIntersect(
            double x, double y, double z,
            double vx, double vy, double vz,
            double& dt
    ) const {
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
