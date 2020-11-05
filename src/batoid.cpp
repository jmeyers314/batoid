#include "batoid.h"

namespace batoid {


    void applyForwardTransformArrays(
        const vec3 dr, const mat3 drot,
        double* x, double* y, double* z,
        size_t n
    ) {
        for(size_t i=0; i<n; i++) {
            double dx = x[i]-dr[0];
            double dy = y[i]-dr[1];
            double dz = z[i]-dr[2];
            x[i] = dx*drot[0] + dy*drot[3] + dz*drot[6];
            y[i] = dx*drot[1] + dy*drot[4] + dz*drot[7];
            z[i] = dx*drot[2] + dy*drot[5] + dz*drot[8];
        }
    }

    void applyReverseTransformArrays(
        const vec3 dr, const mat3 drot,
        double* x, double* y, double* z,
        size_t n
    ) {
        for(size_t i=0; i<n; i++) {
            double xx = x[i]*drot[0] + y[i]*drot[1] + z[i]*drot[2] + dr[0];
            double yy = x[i]*drot[3] + y[i]*drot[4] + z[i]*drot[5] + dr[1];
            double zz = x[i]*drot[6] + y[i]*drot[7] + z[i]*drot[8] + dr[2];
            x[i] = xx;
            y[i] = yy;
            z[i] = zz;
        }
    }

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

        #if defined _OPENMP
            #if _OPENMP >= 201511  // For OpenMP v4.5
                #pragma omp target teams distribute parallel for \
                    map(to:drptr[:3], drotptr[:9])
            #else
                #pragma omp parallel for
            #endif
        #endif
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

        #if defined _OPENMP
            #if _OPENMP >= 201511  // For OpenMP v4.5
                #pragma omp target teams distribute parallel for \
                    map(to:drptr[:3], drotptr[:9])
            #else
                #pragma omp parallel for
            #endif
        #endif
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


    void obscure(const Obscuration& obsc, RayVector& rv) {
        rv.r.syncToDevice();
        rv.vignetted.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        bool* vigptr = rv.vignetted.data;

        const Obscuration* obscDevPtr = obsc.getDevPtr();

        #if defined _OPENMP
            #if _OPENMP >= 201511  // For OpenMP v4.5
                #pragma omp target teams distribute parallel for is_device_ptr(obscDevPtr)
            #else
                #pragma omp parallel for
            #endif
        #endif
        for(int i=0; i<size; i++) {
            vigptr[i] |= obscDevPtr->contains(xptr[i], yptr[i]);
        }
    }


    void intersect(
        const Surface& surface,
        const vec3 dr, const mat3 drot,
        RayVector& rv,
        const Coating* coating
    ) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        rv.t.syncToDevice();
        rv.vignetted.syncToDevice();
        rv.failed.syncToDevice();
        if (coating) {
            rv.wavelength.syncToDevice();
            rv.flux.syncToDevice();
        }
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* tptr = rv.t.data;
        double* wptr = rv.wavelength.data;
        double* fluxptr = rv.flux.data;
        bool* vigptr = rv.vignetted.data;
        bool* failptr = rv.failed.data;

        const Surface* surfaceDevPtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Coating* coatingDevPtr = nullptr;
        if (coating)
            coatingDevPtr = coating->getDevPtr();

        #if defined _OPENMP
            #if _OPENMP >= 201511  // For OpenMP v4.5
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(surfaceDevPtr, coatingDevPtr) \
                    map(to:drptr[:3], drotptr[:9])
            #else
                #pragma omp parallel for
            #endif
        #endif
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
                    if (coatingDevPtr) {
                        double nx, ny, nz;
                        surfaceDevPtr->normal(x, y, nx, ny, nz);
                        double n1 = vx*vx;
                        n1 += vy*vy;
                        n1 += vz*vz;
                        n1 = 1/sqrt(n1);
                        double alpha = vx*nx;
                        alpha += vy*ny;
                        alpha += vz*nz;
                        alpha *= n1;
                        fluxptr[i] *= coatingDevPtr->getTransmit(wptr[i], alpha);
                    }
                } else {
                    failptr[i] = true;
                    vigptr[i] = true;
                }
            }
        }
    }


    void reflect(
        const Surface& surface,
        const vec3 dr, const mat3 drot,
        RayVector& rv,
        const Coating* coating
    ) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        rv.t.syncToDevice();
        rv.vignetted.syncToDevice();
        rv.failed.syncToDevice();
        if (coating) {
            rv.wavelength.syncToDevice();
            rv.flux.syncToDevice();
        }
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* tptr = rv.t.data;
        double* wptr = rv.wavelength.data;
        double* fluxptr = rv.flux.data;
        bool* vigptr = rv.vignetted.data;
        bool* failptr = rv.failed.data;

        const Surface* surfaceDevPtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Coating* coatingDevPtr = nullptr;
        if (coating)
            coatingDevPtr = coating->getDevPtr();

        #if defined _OPENMP
            #if _OPENMP >= 201511  // For OpenMP v4.5
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(surfaceDevPtr, coatingDevPtr) \
                    map(to:drptr[:3], drotptr[:9])
            #else
                #pragma omp parallel for
            #endif
        #endif
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
            if (!failptr[i]) {
                // intersection
                double dt;
                bool success = surfaceDevPtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
                if (success) {
                    // propagation
                    x += vx * dt;
                    y += vy * dt;
                    z += vz * dt;
                    t += dt;
                    // reflection
                    double nx, ny, nz;
                    surfaceDevPtr->normal(x, y, nx, ny, nz);
                    // alpha = v dot normVec
                    double alpha = vx*nx;
                    alpha += vy*ny;
                    alpha += vz*nz;
                    // v -= 2 alpha normVec
                    vx -= 2*alpha*nx;
                    vy -= 2*alpha*ny;
                    vz -= 2*alpha*nz;
                    // output
                    xptr[i] = x;
                    yptr[i] = y;
                    zptr[i] = z;
                    vxptr[i] = vx;
                    vyptr[i] = vy;
                    vzptr[i] = vz;
                    tptr[i] = t;
                    if (coatingDevPtr) {
                        double nx, ny, nz;
                        surfaceDevPtr->normal(x, y, nx, ny, nz);
                        double n1 = vx*vx;
                        n1 += vy*vy;
                        n1 += vz*vz;
                        n1 = 1/sqrt(n1);
                        alpha *= n1;
                        fluxptr[i] *= coatingDevPtr->getReflect(wptr[i], alpha);
                    }
                } else {
                    failptr[i] = true;
                    vigptr[i] = true;
                }
            }
        }
    }


    void refract(
        const Surface& surface,
        const vec3 dr, const mat3 drot,
        const Medium& m1, const Medium& m2,
        RayVector& rv,
        const Coating* coating
    ) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        rv.t.syncToDevice();
        rv.vignetted.syncToDevice();
        rv.failed.syncToDevice();
        rv.wavelength.syncToDevice();
        if (coating) {
            rv.flux.syncToDevice();
        }
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* tptr = rv.t.data;
        double* wptr = rv.wavelength.data;
        double* fluxptr = rv.flux.data;
        bool* vigptr = rv.vignetted.data;
        bool* failptr = rv.failed.data;

        const Surface* surfaceDevPtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Medium* mDevPtr = m2.getDevPtr();
        const Coating* coatingDevPtr = nullptr;
        if (coating)
            coatingDevPtr = coating->getDevPtr();

        #if defined _OPENMP
            #if _OPENMP >= 201511  // For OpenMP v4.5
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(surfaceDevPtr, mDevPtr, coatingDevPtr) \
                    map(to:drptr[:3], drotptr[:9])
            #else
                #pragma omp parallel for
            #endif
        #endif
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
            if (!failptr[i]) {
                // intersection
                double dt;
                bool success = surfaceDevPtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
                if (success) {
                    // propagation
                    x += vx * dt;
                    y += vy * dt;
                    z += vz * dt;
                    t += dt;
                    // refraction
                    // We can get n1 from the velocity, rather than computing through Medium1...
                    double n1 = vx*vx;
                    n1 += vy*vy;
                    n1 += vz*vz;
                    n1 = 1/sqrt(n1);
                    double nvx = vx*n1;
                    double nvy = vy*n1;
                    double nvz = vz*n1;
                    double nx, ny, nz;
                    surfaceDevPtr->normal(x, y, nx, ny, nz);
                    // alpha = v dot normVec
                    double alpha = nvx*nx;
                    alpha += nvy*ny;
                    alpha += nvz*nz;
                    if (alpha > 0.) {
                        nx *= -1;
                        ny *= -1;
                        nz *= -1;
                        alpha *= -1;
                    }
                    double n2 = mDevPtr->getN(wptr[i]);
                    double eta = n1/n2;
                    double sinsqr = eta*eta*(1-alpha*alpha);
                    double nfactor = eta*alpha + sqrt(1-sinsqr);
                    // output
                    vxptr[i] = eta*nvx - nfactor*nx;
                    vyptr[i] = eta*nvy - nfactor*ny;
                    vzptr[i] = eta*nvz - nfactor*nz;
                    vxptr[i] /= n2;
                    vyptr[i] /= n2;
                    vzptr[i] /= n2;
                    xptr[i] = x;
                    yptr[i] = y;
                    zptr[i] = z;
                    tptr[i] = t;
                    if (coatingDevPtr) {
                        fluxptr[i] *= coatingDevPtr->getTransmit(wptr[i], alpha);
                    }
                } else {
                    failptr[i] = true;
                    vigptr[i] = true;
                }
            }
        }
    }

    void rSplit(
        const Surface& surface,
        const vec3 dr, const mat3 drot,
        const Medium& m1, const Medium& m2,
        const Coating& coating,
        RayVector& rv, RayVector& rvSplit
    ) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        rv.t.syncToDevice();
        rv.wavelength.syncToDevice();
        rv.flux.syncToDevice();
        rv.vignetted.syncToDevice();
        rv.failed.syncToDevice();
        rvSplit.r.syncState = SyncState::device;
        rvSplit.v.syncState = SyncState::device;
        rvSplit.t.syncState = SyncState::device;
        rvSplit.wavelength.syncState = SyncState::device;
        rvSplit.flux.syncState = SyncState::device;
        rvSplit.vignetted.syncState = SyncState::device;
        rvSplit.failed.syncState = SyncState::device;

        // Original RayVector will get replaced with refraction
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        double* tptr = rv.t.data;
        double* wptr = rv.wavelength.data;
        double* fluxptr = rv.flux.data;
        bool* vigptr = rv.vignetted.data;
        bool* failptr = rv.failed.data;

        // rvSplit will contain reflection
        double* xptr2 = rvSplit.r.data;
        double* yptr2 = xptr2 + size;
        double* zptr2 = yptr2 + size;
        double* vxptr2 = rvSplit.v.data;
        double* vyptr2 = vxptr2 + size;
        double* vzptr2 = vyptr2 + size;
        double* tptr2 = rvSplit.t.data;
        double* wptr2 = rvSplit.wavelength.data;
        double* fluxptr2 = rvSplit.flux.data;
        bool* vigptr2 = rvSplit.vignetted.data;
        bool* failptr2 = rvSplit.failed.data;

        const Surface* surfaceDevPtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Medium* mDevPtr = m2.getDevPtr();
        const Coating* cDevPtr = coating.getDevPtr();

        #if defined _OPENMP
            #if _OPENMP >= 201511  // For OpenMP v4.5
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(surfaceDevPtr, mDevPtr, cDevPtr) \
                    map(to:drptr[:3], drotptr[:9])
            #else
                #pragma omp parallel for
            #endif
        #endif
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
            if (!failptr[i]) {
                // intersection
                double dt;
                bool success = surfaceDevPtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
                if (success) {
                    // propagation
                    x += vx * dt;
                    y += vy * dt;
                    z += vz * dt;
                    t += dt;

                    // Calculations common to reflect/refract
                    // We can get n1 from the velocity, rather than computing through Medium1...
                    double n1 = vx*vx;
                    n1 += vy*vy;
                    n1 += vz*vz;
                    n1 = 1/sqrt(n1);
                    double nvx = vx*n1;
                    double nvy = vy*n1;
                    double nvz = vz*n1;
                    double nx, ny, nz;
                    surfaceDevPtr->normal(x, y, nx, ny, nz);
                    double alpha = nvx*nx;
                    alpha += nvy*ny;
                    alpha += nvz*nz;
                    if (alpha > 0) {
                        nx *= -1;
                        ny *= -1;
                        nz *= -1;
                        alpha *= -1;
                    }

                    // Flux coefficients
                    double reflect, transmit;
                    cDevPtr->getCoefs(wptr[i], alpha, reflect, transmit);

                    // Reflection
                    xptr2[i] = x;
                    yptr2[i] = y;
                    zptr2[i] = z;
                    vxptr2[i] = vx - 2*alpha*nx/n1;
                    vyptr2[i] = vy - 2*alpha*ny/n1;
                    vzptr2[i] = vz - 2*alpha*nz/n1;
                    tptr2[i] = t;
                    wptr2[i] = wptr[i];
                    fluxptr2[i] = fluxptr[i]*reflect;
                    vigptr2[i] = vigptr[i];
                    failptr2[i] = failptr[i];

                    // refraction
                    double n2 = mDevPtr->getN(wptr[i]);
                    double eta = n1/n2;
                    double sinsqr = eta*eta*(1-alpha*alpha);
                    double nfactor = eta*alpha + sqrt(1-sinsqr);
                    xptr[i] = x;
                    yptr[i] = y;
                    zptr[i] = z;
                    vxptr[i] = eta*nvx - nfactor*nx;
                    vyptr[i] = eta*nvy - nfactor*ny;
                    vzptr[i] = eta*nvz - nfactor*nz;
                    vxptr[i] /= n2;
                    vyptr[i] /= n2;
                    vzptr[i] /= n2;
                    tptr[i] = t;
                    fluxptr[i] *= transmit;
                } else {
                    vigptr[i] = true;
                    failptr[i] = true;
                    vigptr2[i] = true;
                    failptr2[i] = true;
                }
            }
        }
    }
}
