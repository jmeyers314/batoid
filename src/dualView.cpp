#include "dualView.h"
#include <iostream>

namespace batoid {
    template<typename T>
    DualView<T>::DualView(T* _hostData, size_t _size, DVOwnerType _owner, int _dnum, int _hnum) :
        owner(_owner),
        hostData(_hostData),
        size(_size),
        dnum(_dnum),
        hnum(_hnum),
        deviceData(static_cast<T*>(omp_target_alloc(size*sizeof(T), dnum))),
        owns(false)
    { }

    template<typename T>
    DualView<T>::DualView(size_t _size, DVOwnerType _owner, int _dnum, int _hnum) :
        owner(_owner),
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
        if (owner == DVOwnerType::device) {
            omp_target_memcpy(hostData, deviceData, size*sizeof(T), 0, 0, hnum, dnum);
            owner = DVOwnerType::host;
        }
    }

    template<typename T>
    void DualView<T>::syncToDevice() const {
        if (owner == DVOwnerType::host) {
            omp_target_memcpy(deviceData, hostData, size*sizeof(T), 0, 0, dnum, hnum);
            owner = DVOwnerType::device;
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
}
