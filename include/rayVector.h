#ifndef batoid_rayVector_h
#define batoid_rayVector_h

#include <complex>
#include "dualView.h"

namespace batoid {
    struct RayVector {
    public:
        RayVector(
            double* r, double* v, double* t,
            double* wavelength, double* flux,
            bool* vignetted, bool* failed,
            size_t N
        );

        bool operator==(const RayVector& rhs) const;
        bool operator!=(const RayVector& rhs) const;
        void positionAtTime(double t, double* out) const;
        void propagateInPlace(double t);
        void phase(double x, double y, double z, double t, double* out) const;
        void amplitude(double x, double y, double z, double t, std::complex<double>* out) const;
        std::complex<double> sumAmplitude(double x, double y, double z, double t, bool ignoreVignetted=true) const;

        DualView<double> r;           // 24
        DualView<double> v;           // 48
        DualView<double> t;           // 56
        DualView<double> wavelength;  // 64
        DualView<double> flux;        // 72
        DualView<bool> vignetted;     // 73
        DualView<bool> failed;        // 74 cumulative bytes per Ray.
        size_t size;
    };
}

#endif
