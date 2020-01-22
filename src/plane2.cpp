#include "plane2.h"
#include "utils.h"
#include <cmath>


namespace batoid {
    double Plane2::_sag(double, double) const {
        return 0.0;
    }

    #pragma omp declare target
    void Plane2::_normal(double, double, double& nx, double& ny, double& nz) const {
        nx = 0.0;
        ny = 0.0;
        nz = 1.0;
    }

    bool Plane2::_timeToIntersect(double x, double y, double z, double vx, double vy, double vz, double& dt) const {
        dt = -z/vz;
        return (_allowReverse || dt >= 0.0);
    }
    #pragma omp end declare target

    void Plane2::_intersectInPlace(RayVector2& rv) const {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        rv.t.syncToDevice();
        rv.vignetted.syncToDevice();
        rv.failed.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.deviceData;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* tptr = rv.t.deviceData;
        bool* vigptr = rv.vignetted.deviceData;
        bool* failptr = rv.failed.deviceData;
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                if (!failptr[i]) {
                    double dt = -zptr[i]/vzptr[i];
                    if (!_allowReverse && dt < 0) {
                        failptr[i] = true;
                        vigptr[i] = true;
                    } else {
                        xptr[i] += vxptr[i] * dt;
                        yptr[i] += vyptr[i] * dt;
                        zptr[i] += vzptr[i] * dt;
                        tptr[i] += dt;
                    }
                }
            }
        }
    }

    void Plane2::_reflectInPlace(RayVector2& rv) const {
        _intersectInPlace(rv);
        size_t size = rv.size;
        double* vzptr = rv.v.deviceData+2*size;

        #pragma omp target is_device_ptr(vzptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                vzptr[i] *= -1;
            }
        }
    }

    void Plane2::_refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2) const {
        intersectInPlace(rv);
        size_t size = rv.size;
        double* vxptr = rv.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* wptr = rv.wavelength.deviceData;

        // DualView<double> n1(size);
        // double* n1ptr = n1.deviceData;
        // m1.getNMany(rv.wavelength, n1);
        DualView<double> n2(size);
        double* n2ptr = n2.deviceData;
        m2.getNMany(rv.wavelength, n2);

        #pragma omp target is_device_ptr(n2ptr, vxptr, vyptr, vzptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double n1 = vxptr[i]*vxptr[i];
                n1 += vyptr[i]*vyptr[i];
                n1 += vzptr[i]*vzptr[i];
                n1 = 1/sqrt(n1);

                double discriminant = vzptr[i]*vzptr[i] * n1*n1;
                discriminant -= (1-n2ptr[i]*n2ptr[i]/(n1*n1));

                double norm = n1*n1*vxptr[i]*vxptr[i];
                norm += n1*n1*vyptr[i]*vyptr[i];
                norm += discriminant;
                norm = sqrt(norm);
                vxptr[i] = n1*vxptr[i]/norm/n2ptr[i];
                vyptr[i] = n1*vyptr[i]/norm/n2ptr[i];
                vzptr[i] = sqrt(discriminant)/norm/n2ptr[i];
            }
        }
    }

    // Specializations
    template<>
    void Surface2CRTP<Plane2>::intersectInPlace(RayVector2& rv) const {
        const Plane2* self = static_cast<const Plane2*>(this);
        self->_intersectInPlace(rv);
    }

    template<>
    void Surface2CRTP<Plane2>::reflectInPlace(RayVector2& rv) const {
        const Plane2* self = static_cast<const Plane2*>(this);
        self->_reflectInPlace(rv);
    }

    template<>
    void Surface2CRTP<Plane2>::refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2) const {
        const Plane2* self = static_cast<const Plane2*>(this);
        self->_refractInPlace(rv, m1, m2);
    }

}
