#ifndef batoid_rayVector2_h
#define batoid_rayVector2_h

#include <vector>
#include <complex>
#include "dualView.h"

namespace batoid {
    struct RayVector2 {
    public:
        RayVector2(
            double* r, double* v, double* t,
            double* wavelength, double* flux,
            bool* vignetted, bool *failed,
            size_t N
        );

        bool operator==(const RayVector2& rhs) const;
        bool operator!=(const RayVector2& rhs) const;
        void positionAtTime(double t, double* out) const;
        void propagateInPlace(double t);
        void phase(double x, double y, double z, double t, double* out) const;
        void amplitude(double x, double y, double z, double t, std::complex<double>* out) const;
        std::complex<double> sumAmplitude(double x, double y, double z, double t) const;

        DualView<double> r;
        DualView<double> v;
        DualView<double> t;
        DualView<double> wavelength;
        DualView<double> flux;
        DualView<bool> vignetted;
        DualView<bool> failed;
        size_t size;
    };
}

#endif
