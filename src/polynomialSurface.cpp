#include "polynomialSurface.h"

namespace batoid {


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

    PolynomialSurface::PolynomialSurface(
        const double* coefs, const double* coefs_gradx, const double* coefs_grady,
        const double xorigin, const double yorigin,
        size_t xsize, size_t ysize
    ) :
        Surface(),
        _coefs(coefs), _coefs_gradx(coefs_gradx), _coefs_grady(coefs_grady),
        _xorigin(xorigin), _yorigin(yorigin),
        _xsize(xsize), _ysize(ysize)
    {}

    PolynomialSurface::~PolynomialSurface() {
        #if defined(BATOID_GPU)
            if (_devPtr) {
                const size_t size = _xsize * _ysize;
                const size_t sizem1 = (_xsize-1) * (_ysize-1);
                const double* coefs = _coefs;
                const double* coefs_gradx = _coefs_gradx;
                const double* coefs_grady = _coefs_grady;
                #pragma omp target exit data \
                    map(release:coefs[:size], coefs_gradx[:sizem1], coefs_grady[:sizem1])
            }
        #endif
    }

    double PolynomialSurface::sag(double x, double y) const {
        return horner2d(x-_xorigin, y-_yorigin, _coefs, _xsize, _ysize);
    }

    void PolynomialSurface::normal(double x, double y, double& nx, double& ny, double& nz) const {
        // The gradient arrays are always shaped 1 row and column smaller than the coef array.
        // The only exception is when the coef array is 1x1, in which case the gradient array is
        // also 1x1, but filled with a zero (so horner still returns the right result).
        nx = -horner2d(x-_xorigin, y-_yorigin, _coefs_gradx, _xsize-1, _ysize-1);
        ny = -horner2d(x-_xorigin, y-_yorigin, _coefs_grady, _xsize-1, _ysize-1);
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

    const Surface* PolynomialSurface::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Surface* ptr;
                // Allocate arrays on device
                const size_t size = _xsize * _ysize;
                const size_t sizem1 = (_xsize-1) * (_ysize-1);
                const double* coefs = _coefs;
                const double* coefs_gradx = _coefs_gradx;
                const double* coefs_grady = _coefs_grady;
                #pragma omp target enter data \
                    map(to:coefs[:size], coefs_gradx[:sizem1], coefs_grady[:sizem1])
                #pragma omp target map(from:ptr)
                {
                    ptr = new PolynomialSurface(
                        coefs, coefs_gradx, coefs_grady, _xorigin, _yorigin, _xsize, _ysize
                    );
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

}
