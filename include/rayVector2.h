#ifndef batoid_rayVector2_h
#define batoid_rayVector2_h

#include <iostream>
#include <Eigen/Dense>
#include <omp.h>

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
        ) : r(_r), v(_v), t(_t),
            wavelength(_wavelength), flux(_flux),
            vignetted(_vignetted), failed(_failed),
            owner(OwnerType::host),
            _size(t.size()),
            _hnum(omp_get_initial_device()),
    	    _dnum(omp_get_default_device()),
            _dt(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum)))
    	{
            std::cout << "_size = " << _size << '\n';
            std::cout << "_hnum = " << _hnum << '\n';
            std::cout << "_dnum = " << _dnum << '\n';
            std::cout << "_dt = " << _dt << '\n';
        }

        ~RayVector2() {
    	    omp_target_free(_dt, _dnum);
        }

        // copy from device to host, so can be inspected or altered on host.
        // note that even inspecting on host will cause a future device action
        // to copy from host->device, even if nothing gets written.
        void synchronize() {
            if (owner == OwnerType::device) {
                std::cout << "c++ sending to host\n";
                int result = omp_target_memcpy(t.data(), _dt, _size*sizeof(double), 0, 0, _hnum, _dnum);
                std::cout << "omp_target_memcpy result = " << result << '\n';
                owner = OwnerType::host;
            }
        }

        void sendToDevice() {
            if (owner == OwnerType::host) {
                std::cout << "c++ sending to device\n";
                int result = omp_target_memcpy(_dt, t.data(), _size*sizeof(double), 0, 0, _dnum, _hnum);
                std::cout << "omp_target_memcpy result = " << result << '\n';
                owner = OwnerType::device;
            }
        }

        void inspect() {
            std::cout << "size = " << _size << '\n';
      	    std::cout << "_hnum = " << _hnum << '\n';
    	    std::cout << "_dnum = " << _dnum << '\n';
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
    private:
      size_t _size;
      int _dnum; // device index
      int _hnum; // host index
      double* _dt;  // device time array
    };
}

#endif
