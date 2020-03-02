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

    void Plane2::_intersectInPlace(RayVector2& rv, const CoordSys* cs) const {
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
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr) map(to:rot[:9],dr[:3])
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
                    // intersection
                    double dt = -z/vz;
                    if (!_allowReverse && dt < 0) {
                        failptr[i] = true;
                        vigptr[i] = true;
                    } else {
                        // propagation
                        x += vx * dt;
                        y += vy * dt;
                        z += vz * dt;
                        t += dt;
                        // output
                        xptr[i] = x;
                        yptr[i] = y;
                        zptr[i] = z;
                        vxptr[i] = vx;
                        vyptr[i] = vy;
                        vzptr[i] = vz;
                        tptr[i] = t;
                    }
                }
            }
        }
        rv.setCoordSys(CoordSys(*cs));
    }

    void Plane2::_reflectInPlace(RayVector2& rv, const CoordSys* cs) const {
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
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr) map(to:rot[:9],dr[:3])
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
                    // intersection
                    double dt = -z/vz;
                    if (!_allowReverse && dt < 0) {
                        failptr[i] = true;
                        vigptr[i] = true;
                    } else {
                        // propagation
                        x += vx * dt;
                        y += vy * dt;
                        z += vz * dt;
                        t += dt;
                        // output
                        xptr[i] = x;
                        yptr[i] = y;
                        zptr[i] = z;
                        vxptr[i] = vx;
                        vyptr[i] = vy;
                        vzptr[i] = -vz; // reflect!
                        tptr[i] = t;
                    }
                }
            }
        }
        rv.setCoordSys(CoordSys(*cs));
    }

    void Plane2::_refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2, const CoordSys* cs) const {
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

        // DualView<double> n1(size);
        // double* n1ptr = n1.deviceData;
        // m1.getNMany(rv.wavelength, n1);
        DualView<double> n2(size);
        double* n2ptr = n2.deviceData;
        m2.getNMany(rv.wavelength, n2);

        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, n2ptr, tptr, vigptr, failptr) map(to:rot[:9],dr[:3])
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
                    // intersection
                    double dt = -z/vz;
                    if (!_allowReverse && dt < 0) {
                        failptr[i] = true;
                        vigptr[i] = true;
                    } else {
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

                        double discriminant = vz*vz * n1*n1;
                        discriminant -= (1-n2ptr[i]*n2ptr[i]/(n1*n1));

                        double norm = n1*n1*vx*vx;
                        norm += n1*n1*vy*vy;
                        norm += discriminant;
                        norm = sqrt(norm);
                        vxptr[i] = n1*vx/norm/n2ptr[i];
                        vyptr[i] = n1*vy/norm/n2ptr[i];
                        vzptr[i] = sqrt(discriminant)/norm/n2ptr[i];

                        // output
                        xptr[i] = x;
                        yptr[i] = y;
                        zptr[i] = z;
                        tptr[i] = t;
                    }
                }
            }
        }
        rv.setCoordSys(CoordSys(*cs));
    }

    // Specializations
    template<>
    void Surface2CRTP<Plane2>::intersectInPlace(RayVector2& rv, const CoordSys* cs) const {
        const Plane2* self = static_cast<const Plane2*>(this);
        self->_intersectInPlace(rv, cs);
    }

    template<>
    void Surface2CRTP<Plane2>::reflectInPlace(RayVector2& rv, const CoordSys* cs) const {
        const Plane2* self = static_cast<const Plane2*>(this);
        self->_reflectInPlace(rv, cs);
    }

    template<>
    void Surface2CRTP<Plane2>::refractInPlace(RayVector2& rv, const Medium2& m1, const Medium2& m2, const CoordSys* cs) const {
        const Plane2* self = static_cast<const Plane2*>(this);
        self->_refractInPlace(rv, m1, m2, cs);
    }

    template<>
    void Surface2CRTP<Plane2>::sag(const double* xptr, const double* yptr, const size_t size, double* out) const {
        for(int i=0; i<size; i++){
            out[i] = 0.0;
        }
    }

    template<>
    void Surface2CRTP<Plane2>::normal(const double* xptr, const double* yptr, const size_t size, double* out) const {
        for(int i=0; i<2*size; i++) {
            out[i] = 0.0;
        }
        for(int i=2*size; i<3*size; i++) {
            out[i] = 0.0;
        }
    }
}
