#include <iostream>
#include "surface2.h"
#include "plane2.h"

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
                    if (self->_timeToIntersect(
                        xptr[i], yptr[i], zptr[i],
                        vxptr[i], vyptr[i], vzptr[i],
                        dt)
                    ) {
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
                double n = vxptr[i]*vxptr[i];
                n += vyptr[i]*vyptr[i];
                n += vzptr[i]*vzptr[i];
                n = 1/sqrt(n);
                double nvx = vxptr[i]*n;
                double nvy = vyptr[i]*n;
                double nvz = vzptr[i]*n;
                double normalx, normaly, normalz;
                self->_normal(xptr[i], yptr[i], normalx, normaly, normalz);
                double alpha = nvx*normalx;
                alpha += nvy*normaly;
                alpha += nvz*normalz;
                vxptr[i] = nvx - 2*alpha*normalx;
                vyptr[i] = nvy - 2*alpha*normaly;
                vzptr[i] = nvz - 2*alpha*normalz;
                double norm = vxptr[i]*vxptr[i];
                norm += vyptr[i]*vyptr[i];
                norm += vzptr[i]*vzptr[i];
                norm = sqrt(norm);
                vxptr[i] /= norm*n;
                vyptr[i] /= norm*n;
                vzptr[i] /= norm*n;
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
                double discriminant = alpha*alpha - (1. - (n2ptr[i]*n2ptr[i])/(n1*n1));
                discriminant = sqrt(discriminant);
                double k1 = -alpha + discriminant;
                double k2 = -alpha - discriminant;

                double f1x = nvx+k1*normalx;
                double f1y = nvy+k1*normaly;
                double f1z = nvz+k1*normalz;
                double norm1 = f1x*f1x + f1y*f1y + f1z*f1z;
                norm1 = 1/sqrt(norm1);
                f1x *= norm1;
                f1y *= norm1;
                f1z *= norm1;

                double f2x = nvx+k2*normalx;
                double f2y = nvy+k2*normaly;
                double f2z = nvz+k2*normalz;
                double norm2 = f2x*f2x + f2y*f2y + f2z*f2z;
                norm2 = 1/sqrt(norm2);
                f2x *= norm2;
                f2y *= norm2;
                f2z *= norm2;

                double dot1 = f1x*nvx + f1y*nvy + f1z*nvz;
                double dot2 = f2x*nvx + f2y*nvy + f2z*nvz;
                if (dot1 > dot2) {
                    vxptr[i] = f1x/n2ptr[i];
                    vyptr[i] = f1y/n2ptr[i];
                    vzptr[i] = f1z/n2ptr[i];
                } else {
                    vxptr[i] = f2x/n2ptr[i];
                    vyptr[i] = f2y/n2ptr[i];
                    vzptr[i] = f2z/n2ptr[i];
                }
            }
        }
    }

    // Instantiations
    template class Surface2CRTP<Plane2>;
}
