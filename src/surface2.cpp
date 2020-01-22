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
        self->_reflectInPlace(rv);
    }

    template<typename T>
    void Surface2CRTP<T>::refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2) const {
        const T* self = static_cast<const T*>(this);
        self->_refractInPlace(rv, m1, m2);
    }

    // Instantiations
    template class Surface2CRTP<Plane2>;
}
