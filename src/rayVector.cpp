#include "rayVector.h"

namespace batoid {
    RayVector::RayVector(
        double* _x, double* _y, double* _z,
        double* _vx, double* _vy, double* _vz,
        double* _t,
        double* _wavelength, double* _flux,
        bool* _vignetted, bool* _failed,
        size_t _size
    ) :
        x(_x, _size),
        y(_y, _size),
        z(_z, _size),
        vx(_vx, _size),
        vy(_vy, _size),
        vz(_vz, _size),
        t(_t, _size),
        wavelength(_wavelength, _size),
        flux(_flux, _size),
        vignetted(_vignetted, _size),
        failed(_failed, _size),
        size(_size)
    { }

    void RayVector::positionAtTime(
        double _t, double* xout, double* yout, double* zout, int max_threads
    ) const {
        x.syncToDevice();
        y.syncToDevice();
        z.syncToDevice();
        vx.syncToDevice();
        vy.syncToDevice();
        vz.syncToDevice();
        t.syncToDevice();
        double* xptr = x.data;
        double* yptr = y.data;
        double* zptr = z.data;
        double* vxptr = vx.data;
        double* vyptr = vy.data;
        double* vzptr = vz.data;
        double* tptr = t.data;
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                map(from:xout[:size],yout[:size],zout[:size])
        #else
            #pragma omp parallel for num_threads(max_threads)
        #endif
        for(int i=0; i<size; i++) {
            xout[i] = xptr[i] + vxptr[i] * (_t-tptr[i]);
            yout[i] = yptr[i] + vyptr[i] * (_t-tptr[i]);
            zout[i] = zptr[i] + vzptr[i] * (_t-tptr[i]);
        }
    }

    void RayVector::propagateInPlace(double _t, int max_threads) {
        x.syncToDevice();
        y.syncToDevice();
        z.syncToDevice();
        vx.syncToDevice();
        vy.syncToDevice();
        vz.syncToDevice();
        t.syncToDevice();
        double* xptr = x.data;
        double* yptr = y.data;
        double* zptr = z.data;
        double* vxptr = vx.data;
        double* vyptr = vy.data;
        double* vzptr = vz.data;
        double* tptr = t.data;
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for
        #else
            #pragma omp parallel for num_threads(max_threads)
        #endif
        for(int i=0; i<size; i++) {
            xptr[i] += vxptr[i] * (_t - tptr[i]);
            yptr[i] += vyptr[i] * (_t - tptr[i]);
            zptr[i] += vzptr[i] * (_t - tptr[i]);
            tptr[i] = _t;
        }
    }

    void RayVector::phase(
        double _x, double _y, double _z, double _t, double* out, int max_threads
    ) const {
        const double PI = 3.14159265358979323846;
        x.syncToDevice();
        y.syncToDevice();
        z.syncToDevice();
        vx.syncToDevice();
        vy.syncToDevice();
        vz.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();

        double* xptr = x.data;
        double* yptr = y.data;
        double* zptr = z.data;
        double* vxptr = vx.data;
        double* vyptr = vy.data;
        double* vzptr = vz.data;
        double* tptr = t.data;
        double* wptr = wavelength.data;
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for map(from:out[:size])
        #else
            #pragma omp parallel for num_threads(max_threads)
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

    void RayVector::amplitude(
        double _x, double _y, double _z, double _t, std::complex<double>* out, int max_threads
    ) const {
        const double PI = 3.14159265358979323846;
        x.syncToDevice();
        y.syncToDevice();
        z.syncToDevice();
        vx.syncToDevice();
        vy.syncToDevice();
        vz.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();

        // phi = k.(r-r0) - (t-t0)omega
        // k = 2 pi v / lambda |v|^2
        // omega = 2 pi / lambda
        // amplitude = exp(i phi)
        double* xptr = x.data;
        double* yptr = y.data;
        double* zptr = z.data;
        double* vxptr = vx.data;
        double* vyptr = vy.data;
        double* vzptr = vz.data;
        double* tptr = t.data;
        double* wptr = wavelength.data;
        double* outptr = reinterpret_cast<double*>(out);
        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for map(from:outptr[:2*size])
        #else
            #pragma omp parallel for num_threads(max_threads)
        #endif
        for(int i=0; i<size; i++) {
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


    std::complex<double> RayVector::sumAmplitude(
        double _x, double _y, double _z, double _t, bool ignoreVignetted, int max_threads
    ) const {
        const double PI = 3.14159265358979323846;
        x.syncToDevice();
        y.syncToDevice();
        z.syncToDevice();
        vx.syncToDevice();
        vy.syncToDevice();
        vz.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();
        flux.syncToDevice();
        vignetted.syncToDevice();
        failed.syncToDevice();

        // phi = k.(r-r0) - (t-t0)omega
        // k = 2 pi v / lambda |v|^2
        // omega = 2 pi / lambda
        // amplitude = exp(i phi)
        double* xptr = x.data;
        double* yptr = y.data;
        double* zptr = z.data;
        double* vxptr = vx.data;
        double* vyptr = vy.data;
        double* vzptr = vz.data;
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
            #pragma omp parallel for reduction(+:real, imag) num_threads(max_threads)
        #endif
        for(int i=0; i<size; i++) {
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
            x == rhs.x
            && y == rhs.y
            && z == rhs.z
            && vx == rhs.vx
            && vy == rhs.vy
            && vz == rhs.vz
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
