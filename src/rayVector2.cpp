#include "rayVector2.h"
#include <iostream>

namespace batoid {
    RayVector2::RayVector2(
        double* _r, double* _v, double* _t,
        double* _wavelength, double* _flux,
        bool* _vignetted, bool* _failed,
        size_t _size, const CoordSys& coordSys
    ) :
        r(_r, 3*_size),
        v(_v, 3*_size),
        t(_t, _size),
        wavelength(_wavelength, _size),
        flux(_flux, _size),
        vignetted(_vignetted, _size),
        failed(_failed, _size),
        size(_size),
        _coordSys(coordSys)
    { }

    void RayVector2::positionAtTime(double _t, double* out) const {
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        double* rptr = r.deviceData;
        double* vptr = v.deviceData;
        double* tptr = t.deviceData;
        #pragma omp target is_device_ptr(rptr, vptr, tptr) map(from:out[0:3*size])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                int j = i + size;
                int k = i + 2*size;
                out[i] = rptr[i] + vptr[i] * (_t-tptr[i]);
                out[j] = rptr[j] + vptr[j] * (_t-tptr[i]);
                out[k] = rptr[k] + vptr[k] * (_t-tptr[i]);
            }
        }
    }

    void RayVector2::propagateInPlace(double _t) {
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        double* rptr = r.deviceData;
        double* vptr = v.deviceData;
        double* tptr = t.deviceData;
        #pragma omp target is_device_ptr(rptr, vptr, tptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                int j=i+size;
                int k=i+2*size;
                rptr[i] += vptr[i] * (_t-tptr[i]);
                rptr[j] += vptr[j] * (_t-tptr[i]);
                rptr[k] += vptr[k] * (_t-tptr[i]);
                tptr[i] = _t;
            }
        }
    }

    void RayVector2::phase(double _x, double _y, double _z, double _t, double* out) const {
        const double PI = 3.14159265358979323846;
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();

        // phi = k.(r-r0) - (t-t0)omega
        // k = 2 pi v / lambda |v|^2
        // omega = 2 pi / lambda
        double* xptr = r.deviceData;
        double* yptr = r.deviceData+size;
        double* zptr = r.deviceData+2*size;
        double* vxptr = v.deviceData;
        double* vyptr = v.deviceData+size;
        double* vzptr = v.deviceData+2*size;
        double* tptr = t.deviceData;
        double* wptr = wavelength.deviceData;
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, wptr) map(from:out[0:size])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double v2 = vxptr[i]*vxptr[i] + vyptr[i]*vyptr[i] + vzptr[i]*vzptr[i];
                out[i] = (_x-xptr[i])*vxptr[i];
                out[i] += (_y-yptr[i])*vyptr[i];
                out[i] += (_z-zptr[i])*vzptr[i];
                out[i] /= v2;
                out[i] -= _t-tptr[i];
                out[i] *= 2 * PI / wptr[i];
            }
        }
    }

    void RayVector2::amplitude(double _x, double _y, double _z, double _t, std::complex<double>* out) const {
        const double PI = 3.14159265358979323846;
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();

        // phi = k.(r-r0) - (t-t0)omega
        // k = 2 pi v / lambda |v|^2
        // omega = 2 pi / lambda
        double* xptr = r.deviceData;
        double* yptr = r.deviceData+size;
        double* zptr = r.deviceData+2*size;
        double* vxptr = v.deviceData;
        double* vyptr = v.deviceData+size;
        double* vzptr = v.deviceData+2*size;
        double* tptr = t.deviceData;
        double* wptr = wavelength.deviceData;
        double* out_ptr = reinterpret_cast<double*>(out);
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, wptr) map(from:out_ptr[0:2*size])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double v2 = vxptr[i]*vxptr[i] + vyptr[i]*vyptr[i] + vzptr[i]*vzptr[i];
                double phase = (_x-xptr[i])*vxptr[i];
                phase += (_y-yptr[i])*vyptr[i];
                phase += (_z-zptr[i])*vzptr[i];
                phase /= v2;
                phase -= _t-tptr[i];
                phase *= 2 * PI / wptr[i];
                out_ptr[2*i] = std::cos(phase);
                out_ptr[2*i+1] = std::sin(phase);
            }
        }
    }

    std::complex<double> RayVector2::sumAmplitude(double _x, double _y, double _z, double _t) const {
        const double PI = 3.14159265358979323846;
        r.syncToDevice();
        v.syncToDevice();
        t.syncToDevice();
        wavelength.syncToDevice();

        // phi = k.(r-r0) - (t-t0)omega
        // k = 2 pi v / lambda |v|^2
        // omega = 2 pi / lambda
        double* xptr = r.deviceData;
        double* yptr = r.deviceData+size;
        double* zptr = r.deviceData+2*size;
        double* vxptr = v.deviceData;
        double* vyptr = v.deviceData+size;
        double* vzptr = v.deviceData+2*size;
        double* tptr = t.deviceData;
        double* wptr = wavelength.deviceData;
        double real=0;
        double imag=0;
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, wptr) map(tofrom:real,imag)
        {
            #pragma omp teams distribute parallel for reduction(+:real,imag)
            for(int i=0; i<size; i++) {
                double v2 = vxptr[i]*vxptr[i] + vyptr[i]*vyptr[i] + vzptr[i]*vzptr[i];
                double phase = (_x-xptr[i])*vxptr[i];
                phase += (_y-yptr[i])*vyptr[i];
                phase += (_z-zptr[i])*vzptr[i];
                phase /= v2;
                phase -= _t-tptr[i];
                phase *= 2 * PI / wptr[i];
                real += std::cos(phase);
                imag += std::sin(phase);
            }
        }
        return std::complex<double>(real, imag);
    }

    bool RayVector2::operator==(const RayVector2& rhs) const {
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

    bool RayVector2::operator!=(const RayVector2& rhs) const {
        return !(*this == rhs);
    }
}
