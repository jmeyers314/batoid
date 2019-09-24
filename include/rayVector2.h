#ifndef batoid_rayVector2_h
#define batoid_rayVector2_h

#include <Eigen/Dense>
using Eigen::MatrixX3d;
using Eigen::VectorXd;
using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

template<class T>
using Ref = Eigen::Ref<T, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

namespace batoid {
    struct RayVector2 {
    public:
        RayVector2(
            Ref<MatrixX3d> _r, Ref<MatrixX3d> _v, Ref<VectorXd> _t,
            Ref<VectorXd> _wavelength, Ref<VectorXd> _flux,
            Ref<VectorXb> _vignetted, Ref<VectorXb> _failed
        );

        // copy from device to host, so can be inspected or altered on host.
        // note that even inspecting on host will cause a future device action
        // to copy from host->device, even if nothing gets written.
        void synchronize() {
            if (owner == OwnerType::device) {
                // TODO: copy from device -> host
                owner = OwnerType::host;
            }
        }

        void sendToDevice() {
            if (owner == OwnerType::host) {
                // TODO: copy from host -> device
                owner = OwnerType::device;
            }
        }

        // Big Question: do I always want to send every array?  Or will, e.g., only position and t
        // be sufficient sometimes?  Could have flags for each variable instead of one flag for the
        // whole RayVector2?

        Ref<MatrixX3d> r;
        Ref<MatrixX3d> v;
        Ref<VectorXd> t;
        Ref<VectorXd> wavelength;
        Ref<VectorXd> flux;
        Ref<VectorXb> vignetted;
        Ref<VectorXb> failed;
        enum class OwnerType { host, device };
        OwnerType owner;
    };
}

#endif
