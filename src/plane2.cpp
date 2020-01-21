#include "plane2.h"
#include "utils.h"
#include <cmath>

namespace batoid {
    double Plane2::_sag(double, double) const {
        return 0.0;
    }

    void Plane2::_normal(double, double, double& nx, double& ny, double& nz) const {
        nx = 0.0;
        ny = 0.0;
        nz = 1.0;
    }

    #pragma omp declare target
    bool Plane2::_timeToIntersect(double x, double y, double z, double vx, double vy, double vz, double& dt) const {
        dt = -z/vz;
        return (_allowReverse || dt >= 0.0);
    }
    #pragma omp end declare target

    void Plane2::_intersectInPlace(RayVector2& rv2) const {
        rv2.r.syncToDevice();
        rv2.v.syncToDevice();
        rv2.t.syncToDevice();
        rv2.vignetted.syncToDevice();
        rv2.failed.syncToDevice();
        size_t size = rv2.size;
        double* xptr = rv2.r.deviceData;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv2.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* tptr = rv2.t.deviceData;
        bool* vigptr = rv2.vignetted.deviceData;
        bool* failptr = rv2.failed.deviceData;
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

    void Plane2::_reflectInPlace(RayVector2& rv2) const {
        // 1. intersect
        _intersectInPlace(rv2);
        // 2. allocate/compute normal vectors.  For a plane, this is constant (0,0,1).
        // 3. allocate/compute alpha which is used in next two steps.
        // data is already synchronized to device from intersect
        size_t size = rv2.size;
        double* vxptr = rv2.v.deviceData;
        double* vyptr = rv2.v.deviceData+size;
        double* vzptr = rv2.v.deviceData+2*size;

        #pragma omp target is_device_ptr(vxptr, vyptr, vzptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double n = vxptr[i]*vxptr[i];
                n += vyptr[i]*vyptr[i];
                n += vzptr[i]*vzptr[i];
                n = 1.0/sqrt(n);
                double alpha = vzptr[i] * n;

                vxptr[i] = n*vxptr[i];
                vyptr[i] = n*vyptr[i];
                vzptr[i] = n*vzptr[i] - 2*alpha;
                double norm = vxptr[i]*vxptr[i];
                norm += vyptr[i]*vyptr[i];
                norm += vzptr[i]*vzptr[i];
                norm = 1.0/(n*sqrt(norm));
                vxptr[i] *= norm;
                vyptr[i] *= norm;
                vzptr[i] *= norm;
            }
        }
    }

    void solveRefractionQuadratic(
        const DualView<double>& alpha, const DualView<double>& n1, const DualView<double>& n2,
        DualView<double>& k1, DualView<double>& k2
    ) {
        alpha.syncToDevice();
        n1.syncToDevice();
        n2.syncToDevice();
        k1.owner = DVOwnerType::device;
        k2.owner = DVOwnerType::device;

        double* alphaptr = alpha.deviceData;
        double* n1ptr = n1.deviceData;
        double* n2ptr = n2.deviceData;
        double* k1ptr = k1.deviceData;
        double* k2ptr = k2.deviceData;

        size_t size = alpha.size;

        #pragma omp target is_device_ptr(alphaptr, n1ptr, n2ptr, k1ptr, k2ptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double discriminant = alphaptr[i]*alphaptr[i];
                discriminant -= (1-n2ptr[i]*n2ptr[i]/(n1ptr[i]*n1ptr[i]));
                discriminant = sqrt(discriminant);
                k1ptr[i] = -alphaptr[i] - discriminant;
                k2ptr[i] = -alphaptr[i] + discriminant;
            }
        }
    }

    void Plane2::_refractInPlace(RayVector2& rv2, const Medium2& m1, const Medium2& m2) const {
        // 1. intersect
        intersectInPlace(rv2);
        // 2. Allocate for refractive indices, alpha.
        size_t size = rv2.size;
        double* vxptr = rv2.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* wptr = rv2.wavelength.deviceData;

        DualView<double> alpha(size);
        DualView<double> n1(size);
        DualView<double> n2(size);
        double* alphaptr = alpha.deviceData;
        double* n1ptr = n1.deviceData;
        double* n2ptr = n2.deviceData;

        // Calculate refractive indices
        m1.getNMany(rv2.wavelength, n1);
        m2.getNMany(rv2.wavelength, n2);

        // Calculate alpha
        #pragma omp target is_device_ptr(vzptr, n1ptr, alphaptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                alphaptr[i] = vzptr[i] * n1ptr[i];
            }
        }

        // Calculate k's
        DualView<double> k1(size);
        DualView<double> k2(size);
        double* k1ptr = k1.deviceData;
        double* k2ptr = k2.deviceData;

        solveRefractionQuadratic(alpha, n1, n2, k1, k2);

        DualView<double> f1x(size);
        DualView<double> f1y(size);
        DualView<double> f1z(size);
        DualView<double> f2x(size);
        DualView<double> f2y(size);
        DualView<double> f2z(size);

        double* f1xptr = f1x.deviceData;
        double* f1yptr = f1y.deviceData;
        double* f1zptr = f1z.deviceData;
        double* f2xptr = f2x.deviceData;
        double* f2yptr = f2y.deviceData;
        double* f2zptr = f2z.deviceData;

        #pragma omp target is_device_ptr(alphaptr, k1ptr, k2ptr, f1zptr, f2zptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                f1zptr[i] = alphaptr[i] + k1ptr[i];
                f2zptr[i] = alphaptr[i] + k2ptr[i];
            }
        }

        #pragma omp target is_device_ptr(n1ptr, vxptr, vyptr, f1xptr, f1yptr, f1zptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double norm = n1ptr[i]*n1ptr[i]*vxptr[i]*vxptr[i];
                norm += n1ptr[i]*n1ptr[i]*vyptr[i]*vyptr[i];
                norm += f1zptr[i]*f1zptr[i];
                norm = sqrt(norm);
                f1xptr[i] = n1ptr[i]*vxptr[i]/norm;
                f1yptr[i] = n1ptr[i]*vyptr[i]/norm;
                f1zptr[i] /= norm;
            }
        }

        #pragma omp target is_device_ptr(n1ptr, vxptr, vyptr, f2xptr, f2yptr, f2zptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double norm = n1ptr[i]*n1ptr[i]*vxptr[i]*vxptr[i];
                norm += n1ptr[i]*n1ptr[i]*vyptr[i]*vyptr[i];
                norm += f2zptr[i]*f2zptr[i];
                norm = sqrt(norm);
                f2xptr[i] = n1ptr[i]*vxptr[i]/norm;
                f2yptr[i] = n1ptr[i]*vyptr[i]/norm;
                f2zptr[i] /= norm;
            }
        }

        #pragma omp target is_device_ptr(vxptr, vyptr, vzptr, f1xptr, f1yptr, f1zptr, f2xptr, f2yptr, f2zptr, n1ptr, n2ptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                if (f1zptr[i] > f2zptr[i]) {
                    vxptr[i] = f1xptr[i]/n2ptr[i];
                    vyptr[i] = f1yptr[i]/n2ptr[i];
                    vzptr[i] = f1zptr[i]/n2ptr[i];
                } else {
                    vxptr[i] = f2xptr[i]/n2ptr[i];
                    vyptr[i] = f2yptr[i]/n2ptr[i];
                    vzptr[i] = f2zptr[i]/n2ptr[i];
                }
            }
        }
    }
}
