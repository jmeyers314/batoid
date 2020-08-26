#include "batoid.h"

#define PI 3.14159265358979323846264338327950288419716939937510L

namespace batoid {


    void applyForwardTransform(const vec3 dr, const mat3 drot, RayVector& rv) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        const double* drptr = dr.data();
        const double* drotptr = drot.data();

        #pragma omp target teams distribute parallel for \
            map(to:drptr[:3], drotptr[:9])
        for(int i=0; i<size; i++) {
            double dx = xptr[i]-drptr[0];
            double dy = yptr[i]-drptr[1];
            double dz = zptr[i]-drptr[2];
            xptr[i] = dx*drotptr[0] + dy*drotptr[3] + dz*drotptr[6];
            yptr[i] = dx*drotptr[1] + dy*drotptr[4] + dz*drotptr[7];
            zptr[i] = dx*drotptr[2] + dy*drotptr[5] + dz*drotptr[8];
            double vx = vxptr[i]*drotptr[0] + vyptr[i]*drotptr[3] + vzptr[i]*drotptr[6];
            double vy = vxptr[i]*drotptr[1] + vyptr[i]*drotptr[4] + vzptr[i]*drotptr[7];
            double vz = vxptr[i]*drotptr[2] + vyptr[i]*drotptr[5] + vzptr[i]*drotptr[8];
            vxptr[i] = vx;
            vyptr[i] = vy;
            vzptr[i] = vz;
        }
    }


    void applyReverseTransform(const vec3 dr, const mat3 drot, RayVector& rv) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        const double* drptr = dr.data();
        const double* drotptr = drot.data();

        #pragma omp target teams distribute parallel for \
            map(to:drptr[:3], drotptr[:9])
        for(int i=0; i<size; i++) {
            double x = xptr[i]*drotptr[0] + yptr[i]*drotptr[1] + zptr[i]*drotptr[2] + drptr[0];
            double y = xptr[i]*drotptr[3] + yptr[i]*drotptr[4] + zptr[i]*drotptr[5] + drptr[1];
            double z = xptr[i]*drotptr[6] + yptr[i]*drotptr[7] + zptr[i]*drotptr[8] + drptr[2];
            xptr[i] = x;
            yptr[i] = y;
            zptr[i] = z;
            double vx = vxptr[i]*drotptr[0] + vyptr[i]*drotptr[1] + vzptr[i]*drotptr[2];
            double vy = vxptr[i]*drotptr[3] + vyptr[i]*drotptr[4] + vzptr[i]*drotptr[5];
            double vz = vxptr[i]*drotptr[6] + vyptr[i]*drotptr[7] + vzptr[i]*drotptr[8];
            vxptr[i] = vx;
            vyptr[i] = vy;
            vzptr[i] = vz;
        }
    }


    void intersect(const Surface& surface, const vec3 dr, const mat3 drot, RayVector& rv) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        rv.t.syncToDevice();
        rv.vignetted.syncToDevice();
        rv.failed.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* tptr = rv.t.data;
        bool* vigptr = rv.vignetted.data;
        bool* failptr = rv.failed.data;

        Surface* surfaceDevPtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();

        #pragma omp target teams distribute parallel for \
            is_device_ptr(surfaceDevPtr) \
            map(to:drptr[:3], drotptr[:9])
        for(int i=0; i<size; i++) {
            // Coordinate transformation
            double dx = xptr[i]-drptr[0];
            double dy = yptr[i]-drptr[1];
            double dz = zptr[i]-drptr[2];
            double x = dx*drotptr[0] + dy*drotptr[3] + dz*drotptr[6];
            double y = dx*drotptr[1] + dy*drotptr[4] + dz*drotptr[7];
            double z = dx*drotptr[2] + dy*drotptr[5] + dz*drotptr[8];
            double vx = vxptr[i]*drotptr[0] + vyptr[i]*drotptr[3] + vzptr[i]*drotptr[6];
            double vy = vxptr[i]*drotptr[1] + vyptr[i]*drotptr[4] + vzptr[i]*drotptr[7];
            double vz = vxptr[i]*drotptr[2] + vyptr[i]*drotptr[5] + vzptr[i]*drotptr[8];
            double t = tptr[i];
            // intersection
            if (!failptr[i]) {
                double dt;
                bool success = surfaceDevPtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
                if (success) {
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

}
