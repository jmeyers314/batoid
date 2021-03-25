#include "rayVector.h"

namespace batoid {
    RayVector::RayVector(
        double* _r, double* _v, double* _t,
        double* _wavelength, double* _flux,
        bool* _vignetted, bool* _failed,
        size_t _size
    ) :
        r(_r, 3*_size),
        v(_v, 3*_size),
        t(_t, _size),
        wavelength(_wavelength, _size),
        flux(_flux, _size),
        vignetted(_vignetted, _size),
        failed(_failed, _size),
        size(_size)
    { }

    void RayVector::positionAtTime(double _t, double* out) const {
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        double* rptr = r.data;
        double* vptr = v.data;
        double* tptr = t.data;
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for map(from:out[:3*size])
        #else
            #pragma omp parallel for
        #endif
        for(int i=0; i<size; i++) {
            int j = i + size;
            int k = i + 2*size;
            out[i] = rptr[i] + vptr[i] * (_t-tptr[i]);
            out[j] = rptr[j] + vptr[j] * (_t-tptr[i]);
            out[k] = rptr[k] + vptr[k] * (_t-tptr[i]);
        }
    }

    void RayVector::propagateInPlace(double _t) {
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        double* rptr = r.data;
        double* vptr = v.data;
        double* tptr = t.data;
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for
        #else
            #pragma omp parallel for
        #endif
        for(int i=0; i<size; i++) {
            int j = i + size;
            int k = i + 2*size;
            rptr[i] += vptr[i] * (_t-tptr[i]);
            rptr[j] += vptr[j] * (_t-tptr[i]);
            rptr[k] += vptr[k] * (_t-tptr[i]);
            tptr[i] = _t;
        }
    }

    void RayVector::phase(double _x, double _y, double _z, double _t, double* out) const {
        const double PI = 3.14159265358979323846;
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();

        double* xptr = r.data;
        double* yptr = r.data+size;
        double* zptr = r.data+2*size;
        double* vxptr = v.data;
        double* vyptr = v.data+size;
        double* vzptr = v.data+2*size;
        double* tptr = t.data;
        double* wptr = wavelength.data;
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for map(from:out[:size])
        #else
            #pragma omp parallel for
        #endif
        for(int i=0; i<size; i++) {
            // phi = k.(r-r0) - (t-t0)omega
            // k = 2 pi v / lambda |v|^2
            // omega = 2 pi / lambda
            double v2 = vxptr[i]*vxptr[i] + vyptr[i]*vyptr[i] + vzptr[i]*vzptr[i];
            out[i] = (_x-xptr[i])*vxptr[i];
            out[i] += (_y-yptr[i])*vyptr[i];
            out[i] += (_z-zptr[i])*vzptr[i];
            out[i] /= v2;
            out[i] -= _t-tptr[i];
            out[i] *= 2 * PI / wptr[i];
        }
    }

    void RayVector::amplitude(double _x, double _y, double _z, double _t, std::complex<double>* out) const {
        const double PI = 3.14159265358979323846;
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();

        // phi = k.(r-r0) - (t-t0)omega
        // k = 2 pi v / lambda |v|^2
        // omega = 2 pi / lambda
        // amplitude = exp(i phi)
        double* xptr = r.data;
        double* vxptr = v.data;
        double* tptr = t.data;
        double* wptr = wavelength.data;
        double* outptr = reinterpret_cast<double*>(out);
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for map(from:outptr[:2*size])
        #else
            #pragma omp parallel for
        #endif
        for(int i=0; i<size; i++) {
            double* yptr = xptr+size;
            double* zptr = xptr+2*size;
            double* vyptr = vxptr+size;
            double* vzptr = vxptr+2*size;

            double v2 = vxptr[i]*vxptr[i] + vyptr[i]*vyptr[i] + vzptr[i]*vzptr[i];
            double phase = (_x-xptr[i])*vxptr[i];
            phase += (_y-yptr[i])*vyptr[i];
            phase += (_z-zptr[i])*vzptr[i];
            phase /= v2;
            phase -= _t-tptr[i];
            phase *= 2 * PI / wptr[i];
            outptr[2*i] = std::cos(phase);
            outptr[2*i+1] = std::sin(phase);
        }
    }


    std::complex<double> RayVector::sumAmplitude(double _x, double _y, double _z, double _t, bool ignoreVignetted) const {
        const double PI = 3.14159265358979323846;
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();
        flux.syncToDevice();
        vignetted.syncToDevice();
        failed.syncToDevice();

        // phi = k.(r-r0) - (t-t0)omega
        // k = 2 pi v / lambda |v|^2
        // omega = 2 pi / lambda
        // amplitude = exp(i phi)
        double* xptr = r.data;
        double* vxptr = v.data;
        double* tptr = t.data;
        double* wptr = wavelength.data;
        double* fluxptr = flux.data;
        bool* vigptr = vignetted.data;
        bool* failptr = failed.data;
        double real=0;
        double imag=0;
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for reduction(+:real,imag)
        #else
            #pragma omp parallel for reduction(+:real, imag)
        #endif
        for(int i=0; i<size; i++) {
            double* yptr = xptr+size;
            double* zptr = xptr+2*size;
            double* vyptr = vxptr+size;
            double* vzptr = vxptr+2*size;

            double v2 = vxptr[i]*vxptr[i] + vyptr[i]*vyptr[i] + vzptr[i]*vzptr[i];
            double phase = (_x-xptr[i])*vxptr[i];
            phase += (_y-yptr[i])*vyptr[i];
            phase += (_z-zptr[i])*vzptr[i];
            phase /= v2;
            phase -= _t-tptr[i];
            phase *= 2 * PI / wptr[i];

            if (!failptr[i] && !(ignoreVignetted && vigptr[i])) {
                real += std::cos(phase)*fluxptr[i];
                imag += std::sin(phase)*fluxptr[i];
            }
        }
        return std::complex<double>(real, imag);
    }

    bool RayVector::operator==(const RayVector& rhs) const {
        return (
            r == rhs.r
            && v == rhs.v
            && t == rhs.t
            && wavelength == rhs.wavelength
            && flux == rhs.flux
            && vignetted == rhs.vignetted
            && failed == rhs.failed
        );
    }

    bool RayVector::operator!=(const RayVector& rhs) const {
        return !(*this == rhs);
    }
}
