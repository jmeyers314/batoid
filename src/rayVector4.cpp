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
        deviceData(static_cast<T*>(omp_target_alloc(size*sizeof(T), dnum)))
    { }

    template<typename T>
    DualView<T>::~DualView() {
        omp_target_free(deviceData, dnum);
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
        #pragma omp target is_device_ptr(ptr, rhs_ptr) map(tofrom:result) reduction(&:result)
        {
            #pragma omp teams distribute parallel for
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
                out[i]        = rptr[i]        + vptr[i]        * (_t-tptr[i]);
                out[i+size]   = rptr[i+size]   + vptr[i+size]   * (_t-tptr[i]);
                out[i+2*size] = rptr[i+2*size] + vptr[i+2*size] * (_t-tptr[i]);
            }
        }
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
