#ifndef batoid_rayVector3_h
#define batoid_rayVector3_h

#include <Eigen/Dense>
#include <omp.h>

template<typename T>
using Ref = Eigen::Ref<T, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

template<typename T>
using EigenT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

namespace batoid {
    template<typename T>
    struct DualView {
    public:
        DualView(Ref<EigenT<T>> _data);
        ~DualView();

        void copyToHost() const;
        void copyToDevice() const;

        enum class OwnerType{ host, device };
        mutable OwnerType owner;

        mutable Ref<EigenT<T>> array;
        size_t size;

    private:
        int _dnum;  // device index
        int _hnum;  // host index
        T* _deviceData;
    };

    using Eigen::MatrixX3d;
    using Eigen::VectorXd;
    using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    struct RayVector3 {
    public:
        RayVector3(
            Ref<MatrixX3d> _r, Ref<MatrixX3d> _v, Ref<VectorXd> _t,
            Ref<VectorXd> _wavelength, Ref<VectorXd> _flux,
            Ref<VectorXb> _vignetted, Ref<VectorXb> _failed
        );

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
