#include "plane.h"
#include <cmath>

namespace batoid {
    bool Plane::timeToIntersect(const Ray& r, double& t) const {
        t = -r.r[2]/r.v[2] + r.t;
        if (!_allowReverse && t < r.t) return false;
        return true;
    }

    void Plane::intersectInPlace(RayVector4& rv4) const {
        rv4.r.syncToDevice();
        rv4.v.syncToDevice();
        rv4.t.syncToDevice();
        rv4.vignetted.syncToDevice();
        rv4.failed.syncToDevice();
        size_t size = rv4.size;
        double* xptr = rv4.r.deviceData;
        double* yptr = rv4.r.deviceData+size;
        double* zptr = rv4.r.deviceData+2*size;
        double* vxptr = rv4.v.deviceData;
        double* vyptr = rv4.v.deviceData+size;
        double* vzptr = rv4.v.deviceData+2*size;
        double* tptr = rv4.t.deviceData;
        bool* vigptr = rv4.vignetted.deviceData;
        bool* failptr = rv4.failed.deviceData;
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

    void Plane::reflectInPlace(RayVector4& rv4, const Coating* coating) const {
        // 1. intersect
        intersectInPlace(rv4);
        // 2. allocate/compute normal vectors.  For a plane, this is constant (0,0,1).  Need to separate b/c virtual.
        // 3. allocate/compute alpha which is used in next two steps.
        // data is already synchronized to device from intersect
        size_t size = rv4.size;
        double* xptr = rv4.r.deviceData;
        double* yptr = rv4.r.deviceData+size;
        double* zptr = rv4.r.deviceData+2*size;
        double* vxptr = rv4.v.deviceData;
        double* vyptr = rv4.v.deviceData+size;
        double* vzptr = rv4.v.deviceData+2*size;
        double* tptr = rv4.t.deviceData;
        bool* vigptr = rv4.vignetted.deviceData;
        bool* failptr = rv4.failed.deviceData;

        OwningDualView<double> alpha(size);
        OwningDualView<double> n(size);
        double* alphaptr = alpha.deviceData;
        double* nptr = n.deviceData;

        #pragma omp target is_device_ptr(vxptr, vyptr, vzptr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double tmp = vxptr[i]*vxptr[i];
                tmp += vyptr[i]*vyptr[i];
                tmp += vzptr[i]*vzptr[i];
                nptr[i] = 1.0/tmp;
                alphaptr[i] = vzptr[i] * nptr[i];
            }
        }

        // 4. do reflection

        // 5. apply coating reflection.
    }

    // void refractInPlace(RayVector4& rv4, const Medium& m1, const Medium& m2, const Coating*) const {
    //
    // }

}
