#include <iostream>
#include "surface2.h"
#include "plane2.h"
#include "sphere2.h"
#include "paraboloid2.h"

namespace batoid {

    template<typename T>
    double Surface2CRTP<T>::sag(double x, double y) const {
        const T* self = static_cast<const T*>(this);
        return self->_sag(x, y);
    }

    template<typename T>
    void Surface2CRTP<T>::normal(double x, double y, double& nx, double& ny, double& nz) const {
        const T* self = static_cast<const T*>(this);
        self->_normal(x, y, nx, ny, nz);
    }

    template<typename T>
    bool Surface2CRTP<T>::timeToIntersect(double x, double y, double z, double vx, double vy, double vz, double& dt) const {
        const T* self = static_cast<const T*>(this);
        return self->_timeToIntersect(x, y, z, vx, vy, vz, dt);
    }

    template<typename T>
    void Surface2CRTP<T>::intersectInPlace(RayVector2& rv) const {
        const T* self = static_cast<const T*>(this);
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
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr) map(to:self[:1])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                if (!failptr[i]) {
                    double dt;
                    bool success = self->_timeToIntersect(
                        xptr[i], yptr[i], zptr[i],
                        vxptr[i], vyptr[i], vzptr[i],
                        dt
                    );
                    if (success && dt >= 0) {
                        xptr[i] += vxptr[i] * dt;
                        yptr[i] += vyptr[i] * dt;
                        zptr[i] += vzptr[i] * dt;
                        tptr[i] += dt;
                    } else {
                        failptr[i] = true;
                        vigptr[i] = true;
                    }
                }
            }
        }

    }

    template<typename T>
    void Surface2CRTP<T>::reflectInPlace(RayVector2& rv) const {
        const T* self = static_cast<const T*>(this);
        self->intersectInPlace(rv);
        rv.r.syncToDevice();  // should be redundant...
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.deviceData;
        double* yptr = xptr + size;
        double* vxptr = rv.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;

        #pragma omp target is_device_ptr(xptr, yptr, vxptr, vyptr, vzptr) map(to:self[:1])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                // get surface normal vector normVec
                double normalx, normaly, normalz;
                self->_normal(xptr[i], yptr[i], normalx, normaly, normalz);
                // alpha = v dot normVec
                double alpha = vxptr[i]*normalx;
                alpha += vyptr[i]*normaly;
                alpha += vzptr[i]*normalz;
                // v -= 2 alpha normVec
                vxptr[i] -= 2*alpha*normalx;
                vyptr[i] -= 2*alpha*normaly;
                vzptr[i] -= 2*alpha*normalz;
            }
        }
    }

    template<typename T>
    void Surface2CRTP<T>::refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2) const {
        const T* self = static_cast<const T*>(this);
        self->intersectInPlace(rv);
        rv.r.syncToDevice();  // should be redundant...
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.deviceData;
        double* yptr = xptr + size;
        double* vxptr = rv.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        // Note, n1 implicitly defined by rv.v.
        // DualView<double> n1(size);
        // double* n1ptr = n1.deviceData;
        // m1.getNMany(rv.wavelength, n1);
        DualView<double> n2(size);
        double* n2ptr = n2.deviceData;
        m2.getNMany(rv.wavelength, n2);

        #pragma omp target is_device_ptr(xptr, yptr, vxptr, vyptr, vzptr, n2ptr) map(to:self[:1])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double n1 = vxptr[i]*vxptr[i];
                n1 += vyptr[i]*vyptr[i];
                n1 += vzptr[i]*vzptr[i];
                n1 = 1/sqrt(n1);
                double nvx = vxptr[i]*n1;
                double nvy = vyptr[i]*n1;
                double nvz = vzptr[i]*n1;
                double normalx, normaly, normalz;
                self->_normal(xptr[i], yptr[i], normalx, normaly, normalz);
                double alpha = nvx*normalx;
                alpha += nvy*normaly;
                alpha += nvz*normalz;
                if (alpha > 0.) {
                    normalx *= -1;
                    normaly *= -1;
                    normalz *= -1;
                    alpha *= -1;
                }
                double eta = n1/n2ptr[i];
                double sinsqr = eta*eta*(1-alpha*alpha);
                double nfactor = eta*alpha + sqrt(1-sinsqr);

                vxptr[i] = eta*nvx - nfactor*normalx;
                vyptr[i] = eta*nvy - nfactor*normaly;
                vzptr[i] = eta*nvz - nfactor*normalz;
                vxptr[i] /= n2ptr[i];
                vyptr[i] /= n2ptr[i];
                vzptr[i] /= n2ptr[i];
            }
        }
    }

    // Instantiations
    template class Surface2CRTP<Plane2>;
    template class Surface2CRTP<Sphere2>;
    template class Surface2CRTP<Paraboloid2>;
}
