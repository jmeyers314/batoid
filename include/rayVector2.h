#ifndef batoid_rayVector2_h
#define batoid_rayVector2_h

#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

using Eigen::MatrixX3d;
using Eigen::VectorXd;
using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

template<class T>
using Ref = Eigen::Ref<T, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

namespace batoid {
    int runtime_init();

    struct RayVector2 {
    public:
        // Only ctor is from pre-allocated numpy arrays.
        RayVector2(
            Ref<MatrixX3d> _r, Ref<MatrixX3d> _v, Ref<VectorXd> _t,
            Ref<VectorXd> _wavelength, Ref<VectorXd> _flux,
            Ref<VectorXb> _vignetted, Ref<VectorXb> _failed
        );
        ~RayVector2();

        // copy from data from device to host.  The intention is that this will be called in the
        // python layer when data needs to be inspected on the host.  A noop if host already owns
        // the data.
        void synchronize() const;
        // Copy data to device for parallel computation.  A noop if device already owns the data.
        void sendToDevice() const;

        bool operator==(const RayVector2& rhs) const;
        bool operator!=(const RayVector2& rhs) const;

        MatrixX3d positionAtTime(double t) const;
        void positionAtTime(double t, Ref<MatrixX3d>) const;

        mutable Ref<MatrixX3d> r;
        mutable Ref<MatrixX3d> v;
        mutable Ref<VectorXd> t;
        mutable Ref<VectorXd> wavelength;
        mutable Ref<VectorXd> flux;
        mutable Ref<VectorXb> vignetted;
        mutable Ref<VectorXb> failed;
        enum class OwnerType { host, device };
        mutable OwnerType owner;

    private:
        size_t _size; // number of rays represented

        int _dnum; // device index
        int _hnum; // host index

        // pointers to device data
        double* _dx;
        double* _dy;
        double* _dz;
        double* _dvx;
        double* _dvy;
        double* _dvz;
        double* _dt;
        double* _dwavelength;
        double* _dflux;
        bool* _dvignetted;
        bool* _dfailed;
    };
}

#endif
