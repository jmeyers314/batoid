#ifndef batoid_rayVector_h
#define batoid_rayVector_h

#include <complex>
#include "dualView.h"

namespace batoid {
    struct RayVector {
    public:
        RayVector(
            double* x, double* y, double* z,
            double* vx, double* vy, double* vz,
            double* t,
            double* wavelength, double* flux,
            bool* vignetted, bool* failed,
            size_t N
        );

        bool operator==(const RayVector& rhs) const;
        bool operator!=(const RayVector& rhs) const;
        void positionAtTime(double t, double* xout, double* yout, double* zout, int max_threads) const;
        void propagateInPlace(double t, int max_threads);
        void phase(double x, double y, double z, double t, double* out, int max_threads) const;
        void amplitude(double x, double y, double z, double t, std::complex<double>* out, int max_threads) const;
        std::complex<double> sumAmplitude(double x, double y, double z, double t, bool ignoreVignetted, int max_threads) const;

        DualView<double> x;           // 8
        DualView<double> y;           // 16
        DualView<double> z;           // 24
        DualView<double> vx;          // 32
        DualView<double> vy;          // 40
        DualView<double> vz;          // 48
        DualView<double> t;           // 56
        DualView<double> wavelength;  // 64
        DualView<double> flux;        // 72
        DualView<bool> vignetted;     // 73
        DualView<bool> failed;        // 74 cumulative bytes per Ray.
        size_t size;
    };
}

#endif
