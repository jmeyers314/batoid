#include "rayVector2.h"

namespace batoid {
    // Add some arbitrary OMP directive to make sure runtime lib gets loaded
    int runtime_init() {
        double x[64];
        double sum, expected_sum, diff, abs_diff;
        struct timeval start_time, stop_time, elapsed_time;
        int i, rtn;

        for (i=0; i<64; ++i) x[i] = (double)i;
        expected_sum = (64/2.) * (64 - 1.);
        sum = 0.0;
#pragma omp target teams distribute parallel for map(to:x[0:64]) map(tofrom:sum) reduction(+:sum)
        for (i=0; i<64; ++i) sum += x[i];

        /* Calculate absolute value inline to avoid dependency
           on libm which causes compilation issues on x86
           when using Clang compiler and OpenMP target offload */
        diff = sum - expected_sum;
        abs_diff = diff >= 0.0 ? diff : -diff;
        rtn = abs_diff < 1.0E-6 ? 0 : 1;
        return rtn;
    }

    RayVector2::RayVector2(
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
        _dx(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dy(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dz(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dvx(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dvy(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dvz(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dt(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dwavelength(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dflux(static_cast<double*>(omp_target_alloc(_size*sizeof(double), _dnum))),
        _dvignetted(static_cast<bool*>(omp_target_alloc(_size*sizeof(bool), _dnum))),
        _dfailed(static_cast<bool*>(omp_target_alloc(_size*sizeof(bool), _dnum)))
    { }

    RayVector2::~RayVector2() {
	    omp_target_free(_dx, _dnum);
	    omp_target_free(_dy, _dnum);
	    omp_target_free(_dz, _dnum);
	    omp_target_free(_dvx, _dnum);
	    omp_target_free(_dvy, _dnum);
	    omp_target_free(_dvz, _dnum);
	    omp_target_free(_dt, _dnum);
	    omp_target_free(_dwavelength, _dnum);
	    omp_target_free(_dflux, _dnum);
	    omp_target_free(_dvignetted, _dnum);
	    omp_target_free(_dfailed, _dnum);
    }

    void RayVector2::synchronize() const {
        if (owner == OwnerType::device) {
            omp_target_memcpy(r.data(), _dx, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(r.data()+_size, _dy, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(r.data()+2*_size, _dz, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(v.data(), _dvx, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(v.data()+_size, _dvy, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(v.data()+2*_size, _dvz, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(t.data(), _dt, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(wavelength.data(), _dwavelength, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(flux.data(), _dflux, _size*sizeof(double), 0, 0, _hnum, _dnum);
            omp_target_memcpy(vignetted.data(), _dvignetted, _size*sizeof(bool), 0, 0, _hnum, _dnum);
            omp_target_memcpy(failed.data(), _dfailed, _size*sizeof(bool), 0, 0, _hnum, _dnum);
            owner = OwnerType::host;
        }
    }

    void RayVector2::sendToDevice() const {
        if (owner == OwnerType::host) {
            omp_target_memcpy(_dx, r.data(), _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dy, r.data()+_size, _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dz, r.data()+2*_size, _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dvx, v.data(), _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dvy, v.data()+_size, _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dvz, v.data()+2*_size, _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dt, t.data(), _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dwavelength, wavelength.data(), _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dflux, flux.data(), _size*sizeof(double), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dvignetted, vignetted.data(), _size*sizeof(bool), 0, 0, _dnum, _hnum);
            omp_target_memcpy(_dfailed, failed.data(), _size*sizeof(bool), 0, 0, _dnum, _hnum);
            owner = OwnerType::device;
        }
    }

    bool RayVector2::operator==(const RayVector2& rhs) const {
        // For now just synchronize to host to do comparison.
        // In the future, could do comparison on device when both
        // this and rhs owner is device.
        synchronize();
        rhs.synchronize();
        return (
            r == rhs.r &&
            v == rhs.v &&
            t == rhs.t &&
            wavelength == rhs.wavelength &&
            flux == rhs.flux &&
            vignetted == rhs.vignetted &&
            failed == rhs.failed
        );
    }

    bool RayVector2::operator!=(const RayVector2& rhs) const {
        return !(*this == rhs);
    }

    MatrixX3d RayVector2::positionAtTime(double t) const {
        MatrixX3d result(_size, 3);
        positionAtTime(t, result);
        return result;
    }

    void RayVector2::positionAtTime(double t, Ref<MatrixX3d> result) const {
        sendToDevice();
        double* xresult = result.data();
        double* yresult = result.data()+_size;
        double* zresult = result.data()+2*_size;
        double* xptr = _dx;
        double* yptr = _dy;
        double* zptr = _dz;
        double* vxptr = _dvx;
        double* vyptr = _dvy;
        double* vzptr = _dvz;
        double* tptr = _dt;
        #pragma omp target is_device_ptr(xptr, vxptr, tptr) map(from:xresult[0:_size])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<_size; i++) {
                xresult[i] = xptr[i] + vxptr[i]*(t-tptr[i]);
            }
        }
        #pragma omp target is_device_ptr(yptr, vyptr, tptr) map(from:yresult[0:_size])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<_size; i++) {
                yresult[i] = yptr[i] + vyptr[i]*(t-tptr[i]);
            }
        }
        #pragma omp target is_device_ptr(zptr, vzptr, tptr) map(from:zresult[0:_size])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<_size; i++) {
                zresult[i] = zptr[i] + vzptr[i]*(t-tptr[i]);
            }
        }
    }
}
