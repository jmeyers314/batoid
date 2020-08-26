#include "batoid.h"
// #include "ray.h"
// #include "surface.h"
// #include "medium.h"
// #include "utils.h"
// #include "coordsys.h"
#include <cmath>
#include <random>
#include <numeric>

#define PI 3.14159265358979323846264338327950288419716939937510L

namespace batoid {

    void intersect(const Surface& surface, RayVector& rv, CoordTransform* ct) {
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
        vec3 dr;
        mat3 rot;
        if (ct) {
            dr = ct->dr;
            rot = ct->rot;
        } else {
            dr = {0,0,0};
            rot = {1,0,0,  0,1,0,  0,0,1};
        }
        double* drptr = dr.data();
        double* rotptr = rot.data();

        #pragma omp target teams distribute parallel for \
            is_device_ptr(surfaceDevPtr) \
            map(to:drptr[:3], rotptr[:9])
        for(int i=0; i<size; i++) {
            // Coordinate transformation
            double dx = xptr[i]-drptr[0];
            double dy = yptr[i]-drptr[1];
            double dz = zptr[i]-drptr[2];
            double x = dx*rotptr[0] + dy*rotptr[3] + dz*rotptr[6];
            double y = dx*rotptr[1] + dy*rotptr[4] + dz*rotptr[7];
            double z = dx*rotptr[2] + dy*rotptr[5] + dz*rotptr[8];
            double vx = vxptr[i]*rotptr[0] + vyptr[i]*rotptr[3] + vzptr[i]*rotptr[6];
            double vy = vxptr[i]*rotptr[1] + vyptr[i]*rotptr[4] + vzptr[i]*rotptr[7];
            double vz = vxptr[i]*rotptr[2] + vyptr[i]*rotptr[5] + vzptr[i]*rotptr[8];
            double t = tptr[i];
            // intersection
            if (!failptr[i]) {
                double dt;
                bool success = surfaceDevPtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
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

}
