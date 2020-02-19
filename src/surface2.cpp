#include <iostream>
#include "surface2.h"
#include "plane2.h"
#include "sphere2.h"
#include "paraboloid2.h"
#include "quadric2.h"
#include "asphere2.h"
#include "coordtransform2.h"

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
    void Surface2CRTP<T>::intersectInPlace(RayVector2& rv, const CoordSys* cs) const {
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
        if (!cs)
            cs = &rv.getCoordSys();
        CoordTransform2 ct(rv.getCoordSys(), *cs);
        const double* rot = ct.getRot().data();
        const double* dr = ct.getDr().data();
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr) map(to:self[:1]) map(to:rot[:9],dr[:3])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                // Coordinate transformation
                double x = (xptr[i]-dr[0])*rot[0] + (yptr[i]-dr[1])*rot[1] + (zptr[i]-dr[2])*rot[2];
                double y = (xptr[i]-dr[0])*rot[3] + (yptr[i]-dr[1])*rot[4] + (zptr[i]-dr[2])*rot[5];
                double z = (xptr[i]-dr[0])*rot[6] + (yptr[i]-dr[1])*rot[7] + (zptr[i]-dr[2])*rot[8];
                double vx = vxptr[i]*rot[0] + vyptr[i]*rot[1] + vzptr[i]*rot[2];
                double vy = vxptr[i]*rot[3] + vyptr[i]*rot[4] + vzptr[i]*rot[5];
                double vz = vxptr[i]*rot[6] + vyptr[i]*rot[7] + vzptr[i]*rot[8];
                double t = tptr[i];
                // intersection
                if (!failptr[i]) {
                    double dt;
                    bool success = self->_timeToIntersect(x, y, z, vx, vy, vz, dt);
                    if (success && dt >= 0) {
                        x += vx * dt;
                        y += vy * dt;
                        z += vz * dt;
                        t += dt;
                        xptr[i] = x;
                        yptr[i] = y;
                        zptr[i] = z;
                        vxptr[i] = vx;
                        vyptr[i] = vy;
                        vzptr[i] = vz;
                        tptr[i] = t;
                    } else {
                        failptr[i] = true;
                        vigptr[i] = true;
                    }
                }
            }
        }
        rv.setCoordSys(CoordSys(*cs));
    }

    template<typename T>
    void Surface2CRTP<T>::reflectInPlace(RayVector2& rv, const CoordSys* cs) const {
        const T* self = static_cast<const T*>(this);
        rv.r.syncToDevice();  // should be redundant...
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
        if (!cs)
            cs = &rv.getCoordSys();
        CoordTransform2 ct(rv.getCoordSys(), *cs);
        const double* rot = ct.getRot().data();
        const double* dr = ct.getDr().data();

        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr) map(to:self[:1]) map(to:rot[:9],dr[:3])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                // Coordinate transformation
                double x = (xptr[i]-dr[0])*rot[0] + (yptr[i]-dr[1])*rot[1] + (zptr[i]-dr[2])*rot[2];
                double y = (xptr[i]-dr[0])*rot[3] + (yptr[i]-dr[1])*rot[4] + (zptr[i]-dr[2])*rot[5];
                double z = (xptr[i]-dr[0])*rot[6] + (yptr[i]-dr[1])*rot[7] + (zptr[i]-dr[2])*rot[8];
                double vx = vxptr[i]*rot[0] + vyptr[i]*rot[1] + vzptr[i]*rot[2];
                double vy = vxptr[i]*rot[3] + vyptr[i]*rot[4] + vzptr[i]*rot[5];
                double vz = vxptr[i]*rot[6] + vyptr[i]*rot[7] + vzptr[i]*rot[8];
                double t = tptr[i];
                if (!failptr[i]) {
                    double dt;
                    // intersection
                    bool success = self->_timeToIntersect(x, y, z, vx, vy, vz, dt);
                    if (success && dt >= 0) {
                        // propagation
                        x += vx * dt;
                        y += vy * dt;
                        z += vz * dt;
                        t += dt;
                        // reflection
                        // get surface normal vector normVec
                        double normalx, normaly, normalz;
                        self->_normal(x, y, normalx, normaly, normalz);
                        // alpha = v dot normVec
                        double alpha = vx*normalx;
                        alpha += vy*normaly;
                        alpha += vz*normalz;
                        // v -= 2 alpha normVec
                        vx -= 2*alpha*normalx;
                        vy -= 2*alpha*normaly;
                        vz -= 2*alpha*normalz;
                        // output
                        xptr[i] = x;
                        yptr[i] = y;
                        zptr[i] = z;
                        vxptr[i] = vx;
                        vyptr[i] = vy;
                        vzptr[i] = vz;
                        tptr[i] = t;
                    } else {
                        failptr[i] = true;
                        vigptr[i] = true;
                    }
                }
            }
        }
        rv.setCoordSys(CoordSys(*cs));
    }

    template<typename T>
    void Surface2CRTP<T>::refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2, const CoordSys* cs) const {
        const T* self = static_cast<const T*>(this);
        rv.r.syncToDevice();  // should be redundant...
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
        if (!cs)
            cs = &rv.getCoordSys();
        CoordTransform2 ct(rv.getCoordSys(), *cs);
        const double* rot = ct.getRot().data();
        const double* dr = ct.getDr().data();
        // Note, n1 implicitly defined by rv.v.
        // DualView<double> n1(size);
        // double* n1ptr = n1.deviceData;
        // m1.getNMany(rv.wavelength, n1);
        DualView<double> n2(size);
        double* n2ptr = n2.deviceData;
        m2.getNMany(rv.wavelength, n2);

        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, n2ptr, tptr, vigptr, failptr) map(to:self[:1]) map(to:rot[:9],dr[:3])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                // Coordinate transformation
                double x = (xptr[i]-dr[0])*rot[0] + (yptr[i]-dr[1])*rot[1] + (zptr[i]-dr[2])*rot[2];
                double y = (xptr[i]-dr[0])*rot[3] + (yptr[i]-dr[1])*rot[4] + (zptr[i]-dr[2])*rot[5];
                double z = (xptr[i]-dr[0])*rot[6] + (yptr[i]-dr[1])*rot[7] + (zptr[i]-dr[2])*rot[8];
                double vx = vxptr[i]*rot[0] + vyptr[i]*rot[1] + vzptr[i]*rot[2];
                double vy = vxptr[i]*rot[3] + vyptr[i]*rot[4] + vzptr[i]*rot[5];
                double vz = vxptr[i]*rot[6] + vyptr[i]*rot[7] + vzptr[i]*rot[8];
                double t = tptr[i];
                // intersection
                if (!failptr[i]) {
                    double dt;
                    bool success = self->_timeToIntersect(x, y, z, vx, vy, vz, dt);
                    // output
                    if (success && dt >= 0) {
                        // propagation
                        x += vx * dt;
                        y += vy * dt;
                        z += vz * dt;
                        t += dt;
                        // refraction
                        double n1 = vx*vx;
                        n1 += vy*vy;
                        n1 += vz*vz;
                        n1 = 1/sqrt(n1);
                        double nvx = vx*n1;
                        double nvy = vy*n1;
                        double nvz = vz*n1;
                        double normalx, normaly, normalz;
                        self->_normal(x, y, normalx, normaly, normalz);
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
                        // output
                        vxptr[i] = eta*nvx - nfactor*normalx;
                        vyptr[i] = eta*nvy - nfactor*normaly;
                        vzptr[i] = eta*nvz - nfactor*normalz;
                        vxptr[i] /= n2ptr[i];
                        vyptr[i] /= n2ptr[i];
                        vzptr[i] /= n2ptr[i];
                        xptr[i] = x;
                        yptr[i] = y;
                        zptr[i] = z;
                        tptr[i] = t;
                    } else {
                        failptr[i] = true;
                        vigptr[i] = true;
                    }
                }
            }
        }
        rv.setCoordSys(CoordSys(*cs));
    }

    // Instantiations
    template class Surface2CRTP<Plane2>;
    template class Surface2CRTP<Sphere2>;
    template class Surface2CRTP<Paraboloid2>;
    template class Surface2CRTP<Quadric2>;
    template class Surface2CRTP<Asphere2>;
}
