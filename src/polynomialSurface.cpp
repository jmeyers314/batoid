#include "polynomialSurface.h"
#include <omp.h>

namespace batoid {


    ///////////////////////
    // PolynomialSurface //
    ///////////////////////

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    PolynomialSurface::PolynomialSurface(
        const double* coefs, const double* coefs_gradx, const double* coefs_grady,
        size_t xsize, size_t ysize
    ) :
        _coefs(coefs), _coefs_gradx(coefs_gradx), _coefs_grady(coefs_grady),
        _xsize(xsize), _ysize(ysize)
    {}

    PolynomialSurface::~PolynomialSurface() {}

    double PolynomialSurface::sag(double x, double y) const {
        return horner2d(x, y, _coefs, _xsize, _ysize);
    }

    void PolynomialSurface::normal(double x, double y, double& nx, double& ny, double& nz) const {
        // The gradient arrays are always shaped 1 row and column smaller than the coef array.
        // The only exception is when the coef array is 1x1, in which case the gradient array is
        // also 1x1, but filled with a zero (so horner still returns the right result).
        nx = -horner2d(x, y, _coefs_gradx, _xsize-1, _ysize-1);
        ny = -horner2d(x, y, _coefs_grady, _xsize-1, _ysize-1);
        nz = 1./std::sqrt(nx*nx + ny*ny + 1);
        nx *= nz;
        ny *= nz;
    }

    double horner(double x, const double* coefs, size_t n) {
        double result = 0.0;
        for (int i=n-1; i>=0; i--) {
            result *= x;
            result += coefs[i];
        }
        return result;
    }

    double horner2d(double x, double y, const double* coefs, size_t nx, size_t ny) {
        double result = 0.0;
        for (int i=ny-1; i>=0; i--) {
            result *= x;
            result += horner(y, coefs + (i*nx), nx);
        }
        return result;
    }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    /////////////////////////////
    // PolynomialSurfaceHandle //
    /////////////////////////////

    PolynomialSurfaceHandle::PolynomialSurfaceHandle(
        const double* coefs, const double* coefs_gradx, const double* coefs_grady,
        size_t xsize, size_t ysize
    ) :
        SurfaceHandle(),
        _coefs(coefs),
        _coefs_gradx(coefs_gradx),
        _coefs_grady(coefs_grady),
        _xsize(xsize),
        _ysize(ysize)
    {
        _hostPtr = new PolynomialSurface(
            _coefs, _coefs_gradx, _coefs_grady, _xsize, _ysize
        );
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(PolynomialSurface), omp_get_default_device());
            const size_t size = _xsize * _ysize;
            const size_t sizem1 = (_xsize-1) * (_ysize-1);
            const double* cfs = _coefs;
            const double* cfsx = _coefs_gradx;
            const double* cfsy = _coefs_grady;
            #pragma omp target enter data map(to:cfs[:size], cfsx[:sizem1], cfsy[:sizem1])
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) PolynomialSurface(cfs, cfsx, cfsy, _xsize, _ysize);
            }
        #endif
    }

    PolynomialSurfaceHandle::~PolynomialSurfaceHandle() {
        #if defined(BATOID_GPU)
            // We know following is noop, but compiler might not...

            // auto devPtr = static_cast<PolynomialSurface *>(_devicePtr);
            // #pragma omp target is_device_ptr(devPtr)
            // {
            //     devPtr->~PolynomialSurface();
            // }

            const size_t size = _xsize * _ysize;
            const size_t sizem1 = (_xsize-1) * (_ysize-1);
            #pragma omp target exit data \
                map(release:_coefs[:size], _coefs_gradx[:sizem1], _coefs_grady[:sizem1])
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }
}
