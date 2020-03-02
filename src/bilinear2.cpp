#include "bilinear2.h"
#include <cmath>


namespace batoid {
    Bilinear2::Bilinear2(double x0, double y0, double dx, double dy, double* z, size_t size) :
        _x0(x0), _y0(y0), _dx(dx), _dy(dy), _z(z, size*size), _size(size)
        {
            // immediately sync array to device
            _z.syncToDevice();
        }

    #pragma omp declare target
    double Bilinear2::_sag(double x, double y) const {
        int ix = int(std::floor((x-_x0)/_dx));
        int iy = int(std::floor((y-_y0)/_dy));
        double xgrid = _x0 + ix*_dx;
        double ygrid = _y0 + iy*_dy;
        double xfrac = (x - xgrid)/_dx;
        double yfrac = (y - ygrid)/_dy;
        double* zptr = _z.deviceData;
        return (
            zptr[(ix)   + _size * (iy)  ] * (1-xfrac) * (1-yfrac) +
            zptr[(ix+1) + _size * (iy)  ] * (xfrac)   * (1-yfrac) +
            zptr[(ix)   + _size * (iy+1)] * (1-xfrac) * (yfrac) +
            zptr[(ix+1) + _size * (iy+1)] * (xfrac)   * (yfrac)
        );
    }

    void Bilinear2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
        int ix = int(std::floor((x-_x0)/_dx));
        int iy = int(std::floor((y-_y0)/_dy));
        double xgrid = _x0 + ix*_dx;
        double ygrid = _y0 + iy*_dy;
        double xfrac = (x - xgrid)/_dx;
        double yfrac = (y - ygrid)/_dy;
        double* zptr = _z.deviceData;

        double dzdx = (
            zptr[(ix)   + _size * (iy)  ] * (yfrac-1) +
            zptr[(ix+1) + _size * (iy)  ] * (1-yfrac) +
            zptr[(ix)   + _size * (iy+1)] * (-yfrac) +
            zptr[(ix+1) + _size * (iy+1)] * (yfrac)
        )/_dx;
        double dzdy = (
            zptr[(ix)   + _size * (iy)  ] * (xfrac-1) +
            zptr[(ix+1) + _size * (iy)  ] * (-xfrac)  +
            zptr[(ix)   + _size * (iy+1)] * (1-xfrac) +
            zptr[(ix+1) + _size * (iy+1)] * (xfrac)
        )/_dy;

        double norm = 1/std::sqrt(dzdx*dzdx + dzdy*dzdy + 1);
        nx = -dzdx*norm;
        ny = -dzdy*norm;
        nz = norm;
    }

    bool Bilinear2::_timeToIntersect(
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
