#include "rayVector4.h"
#include <iostream>

namespace batoid {
    template<typename T>
    DualView<T>::DualView(T* _hostData, size_t _size, int _dnum, int _hnum) :
        owner(OwnerType::host),
        hostData(_hostData),
        size(_size),
        dnum(_dnum),
        hnum(_hnum),
        deviceData(static_cast<T*>(omp_target_alloc(size*sizeof(T), dnum))),
        owns(false)
    { }

    template<typename T>
    DualView<T>::DualView(size_t _size, int _dnum, int _hnum) :
        owner(OwnerType::device),
        hostData(new T[_size]),
        size(_size),
        dnum(_dnum),
        hnum(_hnum),
        deviceData(static_cast<T*>(omp_target_alloc(size*sizeof(T), dnum))),
        owns(true)
    { }

    template<typename T>
    DualView<T>::~DualView() {
        omp_target_free(deviceData, dnum);
        if (owns) delete[] hostData;
    }

    template<typename T>
    void DualView<T>::syncToHost() const {
        if (owner == OwnerType::device) {
            omp_target_memcpy(hostData, deviceData, size*sizeof(T), 0, 0, hnum, dnum);
            owner = OwnerType::host;
        }
    }

    template<typename T>
    void DualView<T>::syncToDevice() const {
        if (owner == OwnerType::host) {
            omp_target_memcpy(deviceData, hostData, size*sizeof(T), 0, 0, dnum, hnum);
            owner = OwnerType::device;
        }
    }

    template<typename T>
    bool DualView<T>::operator==(const DualView<T>& rhs) const {
        // Compare on the device
        bool result{false};
        syncToDevice();
        rhs.syncToDevice();
        T* ptr = deviceData;
        T* rhs_ptr = rhs.deviceData;
        #pragma omp target is_device_ptr(ptr, rhs_ptr) map(tofrom:result)
        {
            #pragma omp teams distribute parallel for reduction(&:result)
            for(size_t i=0; i<size; i++) result &= ptr[i] == rhs_ptr[i];
        }
        return result;
    }

    template<typename T>
    bool DualView<T>::operator!=(const DualView<T>& rhs) const {
        return !(*this == rhs);
    }

    template class DualView<double>;
    template class DualView<bool>;


    RayVector4::RayVector4(
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

    void RayVector4::positionAtTime(double _t, double* out) const {
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

    void RayVector4::propagateInPlace(double _t) {
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

    void RayVector4::phase(double _x, double _y, double _z, double _t, double* out) const {
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

    void RayVector4::amplitude(double _x, double _y, double _z, double _t, std::complex<double>* out) const {
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

    std::complex<double> RayVector4::sumAmplitude(double _x, double _y, double _z, double _t) const {
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

    bool RayVector4::operator==(const RayVector4& rhs) const {
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

    bool RayVector4::operator!=(const RayVector4& rhs) const {
        return !(*this == rhs);
    }
}
