#ifndef batoid_dualView_h
#define batoid_dualView_h

#include <cstdlib>
#include <omp.h>

#pragma omp requires unified_address

namespace batoid {
    enum class DVOwnerType{ host, device };

    template<typename T>
    struct DualView {
    public:
        // Construct from pre-allocated and owned host memory
        // Allocate but don't fill device memory
        DualView(
            T* _hostData,
            size_t _size,
            DVOwnerType _owner=DVOwnerType::host,
            int _dnum=omp_get_default_device(),
            int _hnum=omp_get_initial_device()
        );
        // Own your own host memory
        DualView(
            size_t _size,
            DVOwnerType _owner=DVOwnerType::device,
            int _dnum=omp_get_default_device(),
            int _hnum=omp_get_initial_device()
        );
        ~DualView();

        void syncToHost() const;
        void syncToDevice() const;

        bool operator==(const DualView<T>& rhs) const;
        bool operator!=(const DualView<T>& rhs) const;

        mutable DVOwnerType owner;

        T* hostData;
        size_t size;
        int dnum;  // device index
        int hnum;  // host index
        T* deviceData;
        bool owns;
    };
}

#endif
