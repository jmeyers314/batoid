#include "rayVector3.h"

namespace batoid {
    template<typename T>
    DualView<T>::DualView(Ref<EigenT<T>> data) :
        owner(OwnerType::host),
        array(data),
        size(array.size()),
        _dnum(omp_get_initial_device()),
        _hnum(omp_get_default_device()),
        _deviceData(static_cast<T*>(omp_target_alloc(size*sizeof(T), _dnum)))
    { }

    template<typename T>
    DualView<T>::~DualView() {
        omp_target_free(_deviceData, _dnum);
    }

    template<typename T>
    void DualView<T>::copyToHost() const {
        if (owner == OwnerType::device) {
            omp_target_memcpy(array.data(), _deviceData, size*sizeof(T), 0, 0, _hnum, _dnum);
            owner = OwnerType::host;
        }
    }

    template<typename T>
    void DualView<T>::copyToDevice() const {
        if (owner == OwnerType::host) {
            omp_target_memcpy(_deviceData, array.data(), size*sizeof(T), 0, 0, _dnum, _hnum);
            owner = OwnerType::device;
        }
    }

    // instantiate for double
    template class DualView<double>;
    template class DualView<bool>;
}
