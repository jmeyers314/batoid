#include "plane2.h"
#include "utils.h"
#include <cmath>

namespace batoid {
    void Plane2::intersectInPlace(RayVector2& rv2) const {
        rv2.r.syncToDevice();
        rv2.v.syncToDevice();
        rv2.t.syncToDevice();
        rv2.vignetted.syncToDevice();
        rv2.failed.syncToDevice();
        size_t size = rv2.size;
        double* xptr = rv2.r.deviceData;
        double* yptr = rv2.r.deviceData+size;
        double* zptr = rv2.r.deviceData+2*size;
        double* vxptr = rv2.v.deviceData;
        double* vyptr = rv2.v.deviceData+size;
        double* vzptr = rv2.v.deviceData+2*size;
        double* tptr = rv2.t.deviceData;
        bool* vigptr = rv2.vignetted.deviceData;
        bool* failptr = rv2.failed.deviceData;
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr, tptr, vigptr, failptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                if (!failptr[i]) {
                    double t = -zptr[i]/vzptr[i] + tptr[i];
                    if (!_allowReverse && t < tptr[i]) {
                        failptr[i] = true;
                        vigptr[i] = true;
                    } else {
                        xptr[i] += vxptr[i] * (t-tptr[i]);
                        yptr[i] += vyptr[i] * (t-tptr[i]);
                        zptr[i] += vzptr[i] * (t-tptr[i]);
                        tptr[i] = t;
                    }
                }
            }
        }
    }

    void Plane2::reflectInPlace(RayVector2& rv2) const {
        // 1. intersect
        intersectInPlace(rv2);
        // 2. allocate/compute normal vectors.  For a plane, this is constant (0,0,1).
        // 3. allocate/compute alpha which is used in next two steps.
        // data is already synchronized to device from intersect
        size_t size = rv2.size;
        double* vxptr = rv2.v.deviceData;
        double* vyptr = rv2.v.deviceData+size;
        double* vzptr = rv2.v.deviceData+2*size;
        double* tptr = rv2.t.deviceData;

        DualView<double> alpha(size);
        DualView<double> n(size);
        double* alphaptr = alpha.deviceData;
        double* nptr = n.deviceData;

        #pragma omp target is_device_ptr(vxptr, vyptr, vzptr, nptr, alphaptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double tmp = vxptr[i]*vxptr[i];
                tmp += vyptr[i]*vyptr[i];
                tmp += vzptr[i]*vzptr[i];
                nptr[i] = 1.0/sqrt(tmp);
                alphaptr[i] = vzptr[i] * nptr[i];
            }
        }
        // 4. do reflection
        #pragma omp target is_device_ptr(vxptr, vyptr, vzptr, nptr, alphaptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                vxptr[i] = nptr[i]*vxptr[i];
                vyptr[i] = nptr[i]*vyptr[i];
                vzptr[i] = nptr[i]*vzptr[i] - 2*alphaptr[i];
                double norm = vxptr[i]*vxptr[i];
                norm += vyptr[i]*vyptr[i];
                norm += vzptr[i]*vzptr[i];
                norm = 1.0/(nptr[i]*sqrt(norm));
                vxptr[i] *= norm;
                vyptr[i] *= norm;
                vzptr[i] *= norm;
            }
        }
    }

    void Plane2::refractInPlace(RayVector2& rv2, const Medium2& m1, const Medium2& m2) const {
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

        // // Calculate alpha
        // #pragma omp target is_device_ptr(vxptr, vyptr, vzptr, n1ptr, alphaptr)
        // {
        //     #pragma omp teams distribute parallel for
        //     for(int i=0; i<size; i++) {
        //         double tmp = vxptr[i]*vxptr[i];
        //         tmp += vyptr[i]*vyptr[i];
        //         tmp += vzptr[i]*vzptr[i];
        //         n1ptr[i] = 1.0/sqrt(tmp);
        //         alphaptr[i] = vzptr[i] * n1ptr[i];
        //     }
        // }

        // // Calculate k's
        // DualView<double> k1(size);
        // DualView<double> k2(size);
        // double* k1ptr = k1.deviceData;
        // double* k2ptr = k2.deviceData;

        // #pragma omp target is_device_ptr(alphaptr, k1ptr, k2ptr, n1ptr, n2ptr)
        // {
        //     #pragma omp teams distribute parallel for
        //     for(int i=0; i<size; i++) {
        //         double a = 1.;
        //         double b = 2*alphaptr[i];
        //         double c = (1. - (n2ptr[i]*n2ptr[i])/(n1ptr[i]*n1ptr[i]));
        //         double k1, k2;
        //         solveQuadratic(a, b, c, k1, k2);
        //         k1ptr[i] = k1;
        //         k2ptr[i] = k2;
        //     }
        // }
    }

}
