#ifndef batoid_dualView_h
#define batoid_dualView_h

#include <cstdlib>  // for size_t

namespace batoid {

    // Enum to track whether data is non-stale on the
    // device or the host.
    enum class SyncState{ host, device };

    // Class to synchronize an array between host and device.
    // Uses openmp pragmas, so becomes a noop when openmp is
    // unavailable.
    template<typename T>
    struct DualView {
    public:
        // Constructor for when host data is already allocated,
        // we just want to point to it.
        DualView(
            T* _data,
            size_t _size
        );

        // Constructor for when we want to freshly allocate
        // host data.
        DualView(
            size_t _size,
            SyncState _syncState=SyncState::device
        );

        ~DualView();

        void syncToHost() const;
        void syncToDevice() const;

        bool operator==(const DualView<T>& rhs) const;
        bool operator!=(const DualView<T>& rhs) const;

        mutable SyncState syncState;
        T* data;
        size_t size;
        bool ownsHostData;
    };
}

#endif
