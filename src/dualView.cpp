#include "dualView.h"

namespace batoid {
    template<typename T>
    DualView<T>::DualView(T* _data, size_t _size) :
        data(_data), size(_size), syncState(SyncState::host), ownsHostData(false) {
            #if defined(BATOID_GPU)
                #pragma omp target enter data map(alloc:data[:size])
            #endif
        }

    template<typename T>
    DualView<T>::DualView(size_t _size, SyncState _syncState) :
        data(new T[_size]), size(_size), syncState(_syncState), ownsHostData(true) {
            #if defined(BATOID_GPU)
                #pragma omp target enter data map(alloc:data[:size])
            #endif
        }

    template<typename T>
    DualView<T>::~DualView() {
        #if defined(BATOID_GPU)
            #pragma omp target exit data map(release:data[:size])
        #endif
        if (ownsHostData) delete[] data;
    }

    template<typename T>
    void DualView<T>::syncToHost() const {
        if (syncState == SyncState::device) {
            #if defined(BATOID_GPU)
                #pragma omp target update from(data[:size])
            #endif
            syncState = SyncState::host;
        }
    }

    template<typename T>
    void DualView<T>::syncToDevice() const {
        if (syncState == SyncState::host) {
            #if defined(BATOID_GPU)
                #pragma omp target update to(data[:size])
            #endif
            syncState = SyncState::device;
        }
    }

    template<typename T>
    bool DualView<T>::operator==(const DualView<T>& rhs) const {
        // If both DualViews are currently synced to Device, do the comparison there.
        // Otherwise, copy to host and do comparison there.
        bool result{true};
        if (syncState == SyncState::host && rhs.syncState == SyncState::host) {
            #pragma omp parallel for reduction(&:result)
            for(size_t i=0; i<size; i++) result &= data[i] == rhs.data[i];
        } else {
            syncToDevice();
            rhs.syncToDevice();
            T* myData = data;
            T* rhsData = rhs.data;
            #if defined(BATOID_GPU)
                #pragma omp target teams distribute parallel for reduction(&:result)
            #else
                #pragma omp parallel for reduction(&:result)
            #endif
            for(size_t i=0; i<size; i++) {
                result &= myData[i] == rhsData[i];
            }
        }
        return result;
    }

    template<typename T>
    bool DualView<T>::operator!=(const DualView<T>& rhs) const {
        return !(*this == rhs);
    }

    // instantiate some versions
    template class DualView<double>;
    template class DualView<bool>;
    template class DualView<int>;

}
