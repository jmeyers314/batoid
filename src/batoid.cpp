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

    void finishParallel(
        const vec3 dr, const mat3 drot, const vec3 vv,
        double* r, size_t n
    ) {
        double* x = r;
        double* y = r + n;
        double* z = r + 2*n;

        double vxlocal = -vv[0]*drot[0] - vv[1]*drot[3] - vv[2]*drot[6];
        double vylocal = -vv[0]*drot[1] - vv[1]*drot[4] - vv[2]*drot[7];
        double vzlocal = -vv[0]*drot[2] - vv[1]*drot[5] - vv[2]*drot[8];

        for(size_t i=0; i<n; i++) {
            // rotate forward
            double dx = x[i]-dr[0];
            double dy = y[i]-dr[1];
            double dz = z[i]-dr[2];
            x[i] = dx*drot[0] + dy*drot[3] + dz*drot[6];
            y[i] = dx*drot[1] + dy*drot[4] + dz*drot[7];
            z[i] = dx*drot[2] + dy*drot[5] + dz*drot[8];
            // intersect
            double dt = -z[i]/vzlocal;
            x[i] += dt*vxlocal;
            y[i] += dt*vylocal;
            z[i] += dt*vzlocal;
            // rotate reverse
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

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                map(to:drptr[:3], drotptr[:9])
        #else
            #pragma omp parallel for
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

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                map(to:drptr[:3], drotptr[:9])
        #else
            #pragma omp parallel for
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

        const Obscuration* obscPtr = obsc.getDevPtr();

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for is_device_ptr(obscPtr)
        #else
            #pragma omp parallel for
        #endif
        for(int i=0; i<size; i++) {
            vigptr[i] |= obscPtr->contains(xptr[i], yptr[i]);
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

        const Surface* surfacePtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Coating* coatingPtr = nullptr;
        if (coating)
            coatingPtr = coating->getDevPtr();

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                is_device_ptr(surfacePtr, coatingPtr) \
                map(to:drptr[:3], drotptr[:9])
        #else
            #pragma omp parallel for
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
                bool success = surfacePtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
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
                    if (coatingPtr) {
                        double nx, ny, nz;
                        surfacePtr->normal(x, y, nx, ny, nz);
                        double n1 = vx*vx;
                        n1 += vy*vy;
                        n1 += vz*vz;
                        n1 = 1/sqrt(n1);
                        double alpha = vx*nx;
                        alpha += vy*ny;
                        alpha += vz*nz;
                        alpha *= n1;
                        fluxptr[i] *= coatingPtr->getTransmit(wptr[i], alpha);
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

        const Surface* surfacePtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Coating* coatingPtr = nullptr;
        if (coating)
            coatingPtr = coating->getDevPtr();

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                is_device_ptr(surfacePtr, coatingPtr) \
                map(to:drptr[:3], drotptr[:9])
        #else
            #pragma omp parallel for
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
                bool success = surfacePtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
                if (success) {
                    // propagation
                    x += vx * dt;
                    y += vy * dt;
                    z += vz * dt;
                    t += dt;
                    // reflection
                    double nx, ny, nz;
                    surfacePtr->normal(x, y, nx, ny, nz);
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
                    if (coatingPtr) {
                        double nx, ny, nz;
                        surfacePtr->normal(x, y, nx, ny, nz);
                        double n1 = vx*vx;
                        n1 += vy*vy;
                        n1 += vz*vz;
                        n1 = 1/sqrt(n1);
                        alpha *= n1;
                        fluxptr[i] *= coatingPtr->getReflect(wptr[i], alpha);
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

        const Surface* surfacePtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Medium* mPtr = m2.getDevPtr();
        const Coating* coatingPtr = nullptr;
        if (coating)
            coatingPtr = coating->getDevPtr();

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                is_device_ptr(surfacePtr, mPtr, coatingPtr) \
                map(to:drptr[:3], drotptr[:9])
        #else
            #pragma omp parallel for
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
                bool success = surfacePtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
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
                    surfacePtr->normal(x, y, nx, ny, nz);
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
                    double n2 = mPtr->getN(wptr[i]);
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
                    if (coatingPtr) {
                        fluxptr[i] *= coatingPtr->getTransmit(wptr[i], alpha);
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

        const Surface* surfacePtr = surface.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();
        const Medium* mPtr = m2.getDevPtr();
        const Coating* cPtr = coating.getDevPtr();

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                is_device_ptr(surfacePtr, mPtr, cPtr) \
                map(to:drptr[:3], drotptr[:9])
        #else
            #pragma omp parallel for
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
                bool success = surfacePtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
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
                    surfacePtr->normal(x, y, nx, ny, nz);
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
                    cPtr->getCoefs(wptr[i], alpha, reflect, transmit);

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
                    double n2 = mPtr->getN(wptr[i]);
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


    void refractScreen(
        const Surface& surface,
        const vec3 dr, const mat3 drot,
        const Surface& screen,
        RayVector& rv
    ) {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        rv.t.syncToDevice();
        rv.vignetted.syncToDevice();
        rv.failed.syncToDevice();
        rv.wavelength.syncToDevice();
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

        const Surface* surfacePtr = surface.getDevPtr();
        const Surface* screenPtr = screen.getDevPtr();
        const double* drptr = dr.data();
        const double* drotptr = drot.data();

        #if defined(BATOID_GPU)
            #pragma omp target teams distribute parallel for \
                is_device_ptr(surfacePtr, screenPtr) \
                map(to:drptr[:3], drotptr[:9])
        #else
            #pragma omp parallel for
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
                bool success = surfacePtr->timeToIntersect(x, y, z, vx, vy, vz, dt);
                if (success) {
                    // propagation
                    x += vx * dt;
                    y += vy * dt;
                    z += vz * dt;
                    t += dt;
                    // screen refraction

                    double norm = std::sqrt(vx*vx + vy*vy + vz*vz);
                    double norm_inv = 1/norm;

                    // printf("vx = %.18f\n", vx);
                    // printf("vy = %.18f\n", vy);
                    // printf("vz = %.18f\n", vz);
                    // printf("|v| = %.18f\n", norm);

                    // Make an orthogonal unit-vector basis:
                    //   e3 is the surface normal
                    double e3x, e3y, e3z;
                    surfacePtr->normal(x, y, e3x, e3y, e3z);

                    //   e1 parallel to y x n
                    double e1norm = std::sqrt(e3z*e3z + e3x*e3x);
                    double e1x = e3z/e1norm;
                    double e1y = 0;
                    double e1z = -e3x/e1norm;

                    //   e2 = e3 x e1
                    double e2x = e3y*e1z - e3z*e1y;
                    double e2y = e3z*e1x - e3x*e1z;
                    double e2z = e3x*e1y - e3y*e1x;

                    // printf("e1x = %.18f\n", e1x);
                    // printf("e1y = %.18f\n", e1y);
                    // printf("e1z = %.18f\n", e1z);

                    // printf("e2x = %.18f\n", e2x);
                    // printf("e2y = %.18f\n", e2y);
                    // printf("e2z = %.18f\n", e2z);

                    // printf("e3x = %.18f\n", e3x);
                    // printf("e3y = %.18f\n", e3y);
                    // printf("e3z = %.18f\n", e3z);

                    // printf("|e1| = %.18f\n", std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z));
                    // printf("|e2| = %.18f\n", std::sqrt(e2x*e2x + e2y*e2y + e2z*e2z));
                    // printf("|e3| = %.18f\n", std::sqrt(e3x*e3x + e3y*e3y + e3z*e3z));

                    // Projections of v onto e1, e2, e3.
                    double cos1 = vx*norm_inv*e1x + vy*norm_inv*e1y + vz*norm_inv*e1z;
                    double cos2 = vx*norm_inv*e2x + vy*norm_inv*e2y + vz*norm_inv*e2z;
                    double cos3 = vx*norm_inv*e3x + vy*norm_inv*e3y + vz*norm_inv*e3z;

                    // printf("cos1 = %.18f\n", cos1);
                    // printf("cos2 = %.18f\n", cos2);
                    // printf("cos3 = %.18f\n", cos3);
                    // printf("len = %.18f\n", std::sqrt(cos1*cos1 + cos2*cos2 + cos3*cos3));

                    // Add screen gradient along e1, e2 to direction cosines
                    double dPdx, dPdy;
                    screenPtr->grad(x, y, dPdx, dPdy);
                    double dPd1 = dPdx*e1x + dPdy*e1y;
                    double dPd2 = dPdx*e2x + dPdy*e2y;

                    // printf("dPd1 = %.18f\n", dPd1);
                    // printf("dPd2 = %.18f\n", dPd2);

                    if (cos3 < 0) {
                        cos1 += dPd1;
                        cos2 += dPd2;
                        cos3 = -std::sqrt(1 - cos1*cos1 - cos2*cos2);
                    } else {
                        cos1 -= dPd1;
                        cos2 -= dPd2;
                        cos3 = std::sqrt(1 - cos1*cos1 - cos2*cos2);
                    }

                    // printf("cos1 = %.18f\n", cos1);
                    // printf("cos2 = %.18f\n", cos2);
                    // printf("cos3 = %.18f\n", cos3);
                    // printf("len = %.18f\n", std::sqrt(cos1*cos1 + cos2*cos2 + cos3*cos3));

                    // Rotate back to xyz.
                    vx = cos1*norm*e1x + cos2*norm*e2x + cos3*norm*e3x;
                    vy = cos1*norm*e1y + cos2*norm*e2y + cos3*norm*e3y;
                    vz = cos1*norm*e1z + cos2*norm*e2z + cos3*norm*e3z;

                    // printf("vx = %.18f\n", vx);
                    // printf("vy = %.18f\n", vy);
                    // printf("vz = %.18f\n", vz);
                    // printf("|v| = %.18f\n", std::sqrt(vx*vx + vy*vy + vz*vz));
                    // printf("\n");

                    t += screenPtr->sag(x, y);

                    // output
                    vxptr[i] = vx;
                    vyptr[i] = vy;
                    vzptr[i] = vz;
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
}
