#include "sphere2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    Sphere2::Sphere2(double R) : _R(R), _Rsq(R*R), _Rinv(1./R), _Rinvsq(1./R/R) {}

    #pragma omp declare target
    double Sphere2::_sag(double x, double y) const {
        if (_R != 0)
            return _R*(1-sqrt(1-(x*x + y*y)*_Rinvsq));
        return 0.0;
    }

    void Sphere2::_normal(double x, double y, double& nx, double& ny, double& nz) const {
        double rsqr = x*x + y*y;
        if (_R == 0.0 || rsqr == 0.0) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        } else {
            double r = sqrt(rsqr);
            double dzdr = _dzdr(r);
            double norm = 1/sqrt(1+dzdr*dzdr);
            nx = -x/r*dzdr*norm;
            ny = -y/r*dzdr*norm;
            nz = norm;
        }
    }

    bool Sphere2::_timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        double vr2 = vx*vx + vy*vy;
        double vz2 = vz*vz;
        double vrr0 = vx*x + vy*y;
        double r02 = x*x + y*y;
        double z0term = z-_R;

        double a = vz2 + vr2;
        double b = 2*vz*z0term + 2*vrr0;
        double c = z0term*z0term - _Rsq + r02;

        double discriminant = b*b - 4*a*c;

        double dt1;
        if (b > 0) {
            dt1 = (-b - sqrt(discriminant)) / (2*a);
        } else {
            dt1 = 2*c / (-b + sqrt(discriminant));
        }
        double dt2 = c / (a*dt1);

        if (dt1 > 0) {
            dt = dt1;
            return true;
        }
        if (dt2 > 0) {
            dt = dt2;
            return true;
        }
        return false;
    }

    double Sphere2::_dzdr(double r) const {
        double rat = r*_Rinv;
        return rat/sqrt(1-rat*rat);
    }
    #pragma omp end declare target

    // void sphere2::_intersectInPlace(RayVector2& rv) const {
    //     rv.r.syncToDevice();
    //     rv.v.syncToDevice();
    //     rv.t.syncToDevice();
    //     rv.vignetted.syncToDevice();
    //     rv.failed.syncToDevice();
    //     size_t size = rv.size;
    //     double* xptr = rv.r.deviceData;
    //     double* yptr = xptr + size;
    //     double* zptr = yptr + size;
    //     double* vxptr = rv.v.deviceData;
    //     double* vyptr = vxptr + size;
    //     double* vzptr = vyptr + size;
    //     double* tptr = rv.t.deviceData;
    //     bool* vigptr = rv.vignetted.deviceData;
    //     bool* failptr = rv.failed.deviceData;
    //     #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr)
    //     {
    //         #pragma omp teams distribute parallel for
    //         for(int i=0; i<size; i++) {
    //             if (!failptr[i]) {
    //                 double dt = -zptr[i]/vzptr[i];
    //                 if (!_allowReverse && dt < 0) {
    //                     failptr[i] = true;
    //                     vigptr[i] = true;
    //                 } else {
    //                     xptr[i] += vxptr[i] * dt;
    //                     yptr[i] += vyptr[i] * dt;
    //                     zptr[i] += vzptr[i] * dt;
    //                     tptr[i] += dt;
    //                 }
    //             }
    //         }
    //     }
    // }
    //
    // void sphere2::_reflectInPlace(RayVector2& rv) const {
    //     _intersectInPlace(rv);
    //     size_t size = rv.size;
    //     double* vzptr = rv.v.deviceData+2*size;
    //
    //     #pragma omp target is_device_ptr(vzptr)
    //     {
    //         #pragma omp teams distribute parallel for
    //         for(int i=0; i<size; i++) {
    //             vzptr[i] *= -1;
    //         }
    //     }
    // }
    //
    // void sphere2::_refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2) const {
    //     intersectInPlace(rv);
    //     size_t size = rv.size;
    //     double* vxptr = rv.v.deviceData;
    //     double* vyptr = vxptr + size;
    //     double* vzptr = vyptr + size;
    //     double* wptr = rv.wavelength.deviceData;
    //
    //     // DualView<double> n1(size);
    //     // double* n1ptr = n1.deviceData;
    //     // m1.getNMany(rv.wavelength, n1);
    //     DualView<double> n2(size);
    //     double* n2ptr = n2.deviceData;
    //     m2.getNMany(rv.wavelength, n2);
    //
    //     #pragma omp target is_device_ptr(n2ptr, vxptr, vyptr, vzptr)
    //     {
    //         #pragma omp teams distribute parallel for
    //         for(int i=0; i<size; i++) {
    //             double n1 = vxptr[i]*vxptr[i];
    //             n1 += vyptr[i]*vyptr[i];
    //             n1 += vzptr[i]*vzptr[i];
    //             n1 = 1/sqrt(n1);
    //
    //             double discriminant = vzptr[i]*vzptr[i] * n1*n1;
    //             discriminant -= (1-n2ptr[i]*n2ptr[i]/(n1*n1));
    //
    //             double norm = n1*n1*vxptr[i]*vxptr[i];
    //             norm += n1*n1*vyptr[i]*vyptr[i];
    //             norm += discriminant;
    //             norm = sqrt(norm);
    //             vxptr[i] = n1*vxptr[i]/norm/n2ptr[i];
    //             vyptr[i] = n1*vyptr[i]/norm/n2ptr[i];
    //             vzptr[i] = sqrt(discriminant)/norm/n2ptr[i];
    //         }
    //     }
    // }

    // // Specializations
    // template<>
    // void Surface2CRTP<sphere2>::intersectInPlace(RayVector2& rv) const {
    //     const sphere2* self = static_cast<const sphere2*>(this);
    //     self->_intersectInPlace(rv);
    // }
    //
    // template<>
    // void Surface2CRTP<sphere2>::reflectInPlace(RayVector2& rv) const {
    //     const sphere2* self = static_cast<const sphere2*>(this);
    //     self->_reflectInPlace(rv);
    // }
    //
    // template<>
    // void Surface2CRTP<sphere2>::refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2) const {
    //     const sphere2* self = static_cast<const sphere2*>(this);
    //     self->_refractInPlace(rv, m1, m2);
    // }

}
