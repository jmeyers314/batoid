#ifndef batoid_rayVector4_h
#define batoid_rayVector4_h

#include <omp.h>
#include <vector>

#pragma omp requires unified_address

namespace batoid {
    template<typename T>
    struct DualView {
    public:
        // Construct from pre-allocated and owned host memory
        // Allocate but don't fill device memory
        DualView(
            T* _hostData,
            size_t _size,
            int _dnum=omp_get_default_device(),
            int _hnum=omp_get_initial_device()
        );
        ~DualView();

        void syncToHost() const;
        void syncToDevice() const;

        bool operator==(const DualView<T>& rhs) const;
        bool operator!=(const DualView<T>& rhs) const;

        enum class OwnerType{ host, device };
        mutable OwnerType owner;

        T* hostData;
        size_t size;
        int dnum;  // device index
        int hnum;  // host index
        T* deviceData;
    };

    template<typename T>
    struct OwningDualView : public DualView<T> {
    public:
        OwningDualView(
            size_t _size,
            int _dnum=omp_get_default_device(),
            int _hnum=omp_get_initial_device()
        );
    private:
        std::vector<T> dataVec;
    };

    struct RayVector4 {
    public:
        RayVector4(
            double* r, double* v, double* t,
            double* wavelength, double* flux,
            bool* vignetted, bool *failed,
            size_t N
        );

        bool operator==(const RayVector4& rhs) const;
        bool operator!=(const RayVector4& rhs) const;
        void positionAtTime(double t, double* out) const;
        void propagateInPlace(double t);

        DualView<double> r;
        DualView<double> v;
        DualView<double> t;
        DualView<double> wavelength;
        DualView<double> flux;
        DualView<bool> vignetted;
        DualView<bool> failed;
        size_t size;
    };
}

#endif
