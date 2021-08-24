#include "table.h"
#include <cmath>


namespace batoid {


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    Table::Table(
        double x0, double y0, double dx, double dy,
        const double* z, const double* dzdx, const double* dzdy, const double* d2zdxdy,
        size_t nx, size_t ny
    ) :
        _devPtr(nullptr),
        _x0(x0), _y0(y0), _dx(dx), _dy(dy),
        _z(z),
        _dzdx(dzdx),
        _dzdy(dzdy),
        _d2zdxdy(d2zdxdy),
        _nx(nx), _ny(ny)
    {}

    Table::~Table() {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                Table* ptr = _devPtr;
                #pragma omp target is_device_ptr(ptr)
                {
                    delete ptr;
                }

                const size_t size = _nx * _ny;
                const double* z = _z;
                const double* dzdx = _dzdx;
                const double* dzdy = _dzdy;
                const double* d2zdxdy = _d2zdxdy;
                #pragma omp target exit data \
                    map(release:z[:size], dzdx[:size], dzdy[:size], d2zdxdy[:size])
            }
        #endif
    }

    double oneDSpline(double x, double val0, double val1, double der0, double der1) {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        double d = val0;

        return d + x*(c + x*(b + x*a));
    }

    double oneDGrad(double x, double val0, double val1, double der0, double der1) {
        double a = 2*(val0-val1) + der0 + der1;
        double b = 3*(val1-val0) - 2*der0 - der1;
        double c = der0;
        return c + x*(2*b + x*3*a);
    }

    double Table::eval(double x, double y) const {
        int ix = int(std::floor((x-_x0)/_dx));
        int iy = int(std::floor((y-_y0)/_dy));
        if ((ix >= _nx) or (ix < 0) or (iy >= _ny) or (iy < 0)) {
            return NAN;
        }
        double xgrid = _x0 + ix*_dx;
        double ygrid = _y0 + iy*_dy;
        double xfrac = (x - xgrid)/_dx;
        double yfrac = (y - ygrid)/_dy;

        double val0 = oneDSpline(
            xfrac,
               _z[_nx*iy + ix],        _z[_nx*iy + ix+1],
            _dzdx[_nx*iy + ix]*_dx, _dzdx[_nx*iy + ix+1]*_dx
        );
        double val1 = oneDSpline(
            xfrac,
               _z[_nx*(iy+1) + ix],        _z[_nx*(iy+1) + ix+1],
            _dzdx[_nx*(iy+1) + ix]*_dx, _dzdx[_nx*(iy+1) + ix+1]*_dx
        );
        double der0 = oneDSpline(
            xfrac,
               _dzdy[_nx*iy + ix],        _dzdy[_nx*iy + ix+1],
            _d2zdxdy[_nx*iy + ix]*_dx, _d2zdxdy[_nx*iy + ix+1]*_dx
        );
        double der1 = oneDSpline(
            xfrac,
               _dzdy[_nx*(iy+1) + ix],        _dzdy[_nx*(iy+1) + ix+1],
            _d2zdxdy[_nx*(iy+1) + ix]*_dx, _d2zdxdy[_nx*(iy+1) + ix+1]*_dx
        );
        return oneDSpline(yfrac, val0, val1, der0*_dy, der1*_dy);
    }

    void Table::grad(
        double x, double y,
        double& dzdx, double& dzdy
    ) const {
        int ix = int(std::floor((x-_x0)/_dx));
        int iy = int(std::floor((y-_y0)/_dy));
        if ((ix >= _nx) or (ix < 0) or (iy >= _ny) or (iy < 0)) {
            dzdx = NAN;
            dzdy = NAN;
            return;
        }
        double xgrid = _x0 + ix*_dx;
        double ygrid = _y0 + iy*_dy;
        double xfrac = (x - xgrid)/_dx;
        double yfrac = (y - ygrid)/_dy;

        // x-gradient
        double val0 = oneDGrad(
            xfrac,
               _z[_nx*iy + ix],        _z[_nx*iy + ix+1],
            _dzdx[_nx*iy + ix]*_dx, _dzdx[_nx*iy + ix+1]*_dx
        );
        double val1 = oneDGrad(
            xfrac,
               _z[_nx*(iy+1) + ix],        _z[_nx*(iy+1) + ix+1],
            _dzdx[_nx*(iy+1) + ix]*_dx, _dzdx[_nx*(iy+1) + ix+1]*_dx
        );
        double der0 = oneDGrad(
            xfrac,
               _dzdy[_nx*iy + ix],        _dzdy[_nx*iy + ix+1],
            _d2zdxdy[_nx*iy + ix]*_dx, _d2zdxdy[_nx*iy + ix+1]*_dx
        );
        double der1 = oneDGrad(
            xfrac,
               _dzdy[_nx*(iy+1) + ix],        _dzdy[_nx*(iy+1) + ix+1],
            _d2zdxdy[_nx*(iy+1) + ix]*_dx, _d2zdxdy[_nx*(iy+1) + ix+1]*_dx
        );
        dzdx = oneDSpline(yfrac, val0, val1, der0*_dy, der1*_dy)/_dx;

        // y-gradient
        val0 = oneDGrad(
            yfrac,
               _z[_nx*iy + ix],        _z[_nx*(iy+1) + ix],
            _dzdy[_nx*iy + ix]*_dy, _dzdy[_nx*(iy+1) + ix]*_dy
        );
        val1 = oneDGrad(
            yfrac,
               _z[_nx*iy + ix+1],        _z[_nx*(iy+1) + ix+1],
            _dzdy[_nx*iy + ix+1]*_dy, _dzdy[_nx*(iy+1) + ix+1]*_dy
        );
        der0 = oneDGrad(
            yfrac,
               _dzdx[_nx*iy + ix],        _dzdx[_nx*(iy+1) + ix],
            _d2zdxdy[_nx*iy + ix]*_dy, _d2zdxdy[_nx*(iy+1) + ix]*_dy
        );
        der1 = oneDGrad(
            yfrac,
               _dzdx[_nx*iy + ix+1],        _dzdx[_nx*(iy+1) + ix+1],
            _d2zdxdy[_nx*iy + ix+1]*_dy, _d2zdxdy[_nx*(iy+1) + ix+1]*_dy
        );
        dzdy = oneDSpline(xfrac, val0, val1, der0*_dx, der1*_dx)/_dy;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Table* Table::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Table* ptr;
                // Allocate arrays on device
                const size_t size = _nx*_ny;
                const double* z = _z;
                const double* dzdx = _dzdx;
                const double* dzdy = _dzdy;
                const double* d2zdxdy = _d2zdxdy;
                #pragma omp target enter data map(to:z[:size], dzdx[:size], dzdy[:size], d2zdxdy[:size])
                #pragma omp target map(from:ptr)
                {
                    ptr = new Table(_x0, _y0, _dx, _dy, z, dzdx, dzdy, d2zdxdy, _nx, _ny);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }
}
