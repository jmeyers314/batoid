#include "surface.h"
#include "coordTransform.h"

namespace batoid {

    bool Surface::timeToIntersect(
        double x, double y, double z,
        double vx, double vy, double vz,
        double& dt
    ) const {
        // The better the initial estimate of dt, the better this will perform
        double rPx = x+vx*dt;
        double rPy = y+vy*dt;
        double rPz = z+vz*dt;

        double sz = sag(rPx, rPy);
        for (int iter=0; iter<10; iter++) {
            // intersect plane tangent to surface at (rPx, rPy, sz)
            // use Newton-Raphson iteration to refine
            double nx, ny, nz;
            normal(rPx, rPy, nx, ny, nz);
            dt = (rPx-x)*nx + (rPy-y)*ny + (sz-z)*nz;
            dt /= (nx*vx + ny*vy + nz*vz);
            rPx = x+vx*dt;
            rPy = y+vy*dt;
            rPz = z+vz*dt;
            sz = sag(rPx, rPy);
            if (std::abs(sz-rPz) < 1e-14) return true;
        }
        return false;
    }

    void Surface::intersectInPlace(RayVector& rv, const CoordSys* cs) const {
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
        if (!cs)
            cs = &rv.getCoordSys();
        CoordTransform transform(rv.getCoordSys(), *cs);
        const double* drptr = transform.dr.data();
        const double* rotptr = transform.rot.data();
        Surface* surfaceDevPtr = getDevPtr();

        #pragma omp target teams distribute parallel for \
            is_device_ptr(surfaceDevPtr) \
            map(to: drptr[:3], rotptr[:9])
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
        rv.setCoordSys(CoordSys(*cs));
    }





    // //
    // // RayVector methods
    // //
    //
    // RayVector Surface::intersect(const RayVector& rv, const CoordSys* cs) const {
    //     // if cs is nullptr, then assume rays and surface are in same coordsys
    //     // otherwise, transform rays into cs coordsys first.
    //     std::vector<Ray> rays(rv.size());
    //     if (!cs)
    //         cs = &rv.getCoordSys();
    //     CoordTransform ct(rv.getCoordSys(), *cs);
    //     parallelTransform(rv.cbegin(), rv.cend(), rays.begin(),
    //         [this, ct](const Ray& r)
    //         {
    //             return _justIntersect(ct.applyForward(r));
    //         }
    //     );
    //     return RayVector(std::move(rays), *cs, rv.getWavelength());
    // }
    //
    // void Surface::intersectInPlace(RayVector& rv, const CoordSys* cs) const {
    //     if (!cs)
    //         cs = &rv.getCoordSys();
    //     CoordTransform ct(rv.getCoordSys(), *cs);
    //     parallel_for_each(
    //         rv.begin(), rv.end(),
    //         [this, ct](Ray& r) {
    //             ct.applyForwardInPlace(r);
    //             _justIntersectInPlace(r);
    //         }
    //     );
    //     rv.setCoordSys(*cs);
    // }
    //
    // RayVector Surface::reflect(const RayVector& rv, const Coating* coating, const CoordSys* cs) const {
    //     std::vector<Ray> rays(rv.size());
    //     if (!cs)
    //         cs = &rv.getCoordSys();
    //     CoordTransform ct(rv.getCoordSys(), *cs);
    //     parallelTransform(
    //         rv.cbegin(), rv.cend(), rays.begin(),
    //         [this,ct,coating](const Ray& r){
    //             double alpha;
    //             Ray out(_justReflect(_justIntersect(ct.applyForward(r)), alpha));
    //             if (coating)
    //                 out.flux *= coating->getReflect(out.wavelength, alpha/out.v.norm());
    //             return out;
    //         }
    //     );
    //     return RayVector(std::move(rays), *cs, rv.getWavelength());
    // }
    //
    // void Surface::reflectInPlace(RayVector& rv, const Coating* coating, const CoordSys* cs) const {
    //     if (!cs)
    //         cs = &rv.getCoordSys();
    //     CoordTransform ct(rv.getCoordSys(), *cs);
    //     parallel_for_each(
    //         rv.begin(), rv.end(),
    //         [this,ct,coating](Ray& r) {
    //             ct.applyForwardInPlace(r);
    //             _justIntersectInPlace(r);
    //             double alpha;
    //             _justReflectInPlace(r, alpha);
    //             if (coating)
    //                 r.flux *= coating->getReflect(r.wavelength, alpha/r.v.norm());
    //         }
    //     );
    //     rv.setCoordSys(*cs);
    // }
    //
    // RayVector Surface::refract(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating* coating, const CoordSys* cs) const {
    //     std::vector<Ray> rays(rv.size());
    //     if (!cs)
    //         cs = &rv.getCoordSys();
    //     CoordTransform ct(rv.getCoordSys(), *cs);
    //     if (std::isnan(rv.getWavelength())) {
    //         parallelTransform(
    //             rv.cbegin(), rv.cend(), rays.begin(),
    //             [this,ct,&m1,&m2,coating](const Ray& r){
    //                 double alpha;
    //                 Ray out(_justRefract(_justIntersect(ct.applyForward(r)), m1, m2, alpha));
    //                 if (coating)
    //                     out.flux *= coating->getTransmit(out.wavelength, alpha/out.v.norm());
    //                 return out;
    //             }
    //         );
    //     } else {
    //         double n1 = m1.getN(rv.getWavelength());
    //         double n2 = m2.getN(rv.getWavelength());
    //         parallelTransform(
    //             rv.cbegin(), rv.cend(), rays.begin(),
    //             [this,ct,n1,n2,coating](const Ray& r){
    //                 double alpha;
    //                 Ray out(_justRefract(_justIntersect(ct.applyForward(r)), n1, n2, alpha));
    //                 if (coating)
    //                     out.flux *= coating->getTransmit(out.wavelength, alpha/out.v.norm());
    //                 return out;
    //             }
    //         );
    //     }
    //     return RayVector(std::move(rays), *cs, rv.getWavelength());
    // }
    //
    // void Surface::refractInPlace(RayVector& rv, const Medium& m1, const Medium& m2, const Coating* coating, const CoordSys* cs) const {
    //     if (!cs)
    //         cs = &rv.getCoordSys();
    //     CoordTransform ct(rv.getCoordSys(), *cs);
    //     if (std::isnan(rv.getWavelength())) {
    //         parallel_for_each(
    //             rv.begin(), rv.end(),
    //             [this,&m1,&m2,coating,ct](Ray& r) {
    //                 double alpha;
    //                 ct.applyForwardInPlace(r);
    //                 _justIntersectInPlace(r);
    //                 _justRefractInPlace(r, m1, m2, alpha);
    //                 if (coating)
    //                     r.flux *= coating->getTransmit(r.wavelength, alpha/r.v.norm());
    //             }
    //         );
    //     } else {
    //         double n1 = m1.getN(rv.getWavelength());
    //         double n2 = m2.getN(rv.getWavelength());
    //         parallel_for_each(
    //             rv.begin(), rv.end(),
    //             [this,n1,n2,coating,ct](Ray& r) {
    //                 double alpha;
    //                 ct.applyForwardInPlace(r);
    //                 _justIntersectInPlace(r);
    //                 _justRefractInPlace(r, n1, n2, alpha);
    //                 if (coating)
    //                     r.flux *= coating->getTransmit(r.wavelength, alpha/r.v.norm());
    //             }
    //         );
    //     }
    //     rv.setCoordSys(*cs);
    // }
    //
    // std::pair<RayVector, RayVector> Surface::rSplit(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating& coating, const CoordSys* cs) const {
    //     RayVector reflected(rv);
    //     RayVector refracted(rv);
    //
    //     reflectInPlace(reflected, &coating, cs);
    //     refractInPlace(refracted, m1, m2, &coating, cs);
    //
    //     return std::make_pair(reflected, refracted);
    // }
    //
    // // std::pair<RayVector, RayVector> Surface::rSplitProb(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating& coating, const CoordTransform* ct) const {
    // //     RayVector reflected(rv);
    // //     RayVector refracted(rv);
    // //
    // //     reflectInPlace(reflected, nullptr, ct);
    // //     refractInPlace(refracted, m1, m2, nullptr, ct);
    // //
    // //     // Go through and probabilistically accept/reject each ray?
    // //     double reflect, transmit, alpha, ran;
    // //     for(unsigned int i=0; i<rv.size(); i++) {
    // //         // Need to recompute the normal vector and alpha=cos(theta)...  for the third time...
    // //         Vector3d normVec(normal(rv[i].r[0], rv[i].r[1]));
    // //         alpha = rv[i].v.normalized().dot(normVec);
    // //         coating.getCoefs(rv[i].wavelength, alpha, reflect, transmit);
    // //         ran = std::uniform_real_distribution<>(0.0, 1.0)(rng);
    // //         if (ran < reflect) { //choose reflect
    // //             refracted[i].vignetted=true;
    // //         } else if (ran < reflect+transmit) { // choose refract
    // //             reflected[i].vignetted=true;
    // //         } else { // choose neither
    // //             refracted[i].vignetted=true;
    // //             reflected[i].vignetted=true;
    // //         }
    // //     }
    // //     reflected.trimVignettedInPlace(0.0);
    // //     refracted.trimVignettedInPlace(0.0);
    // //     return std::make_pair(reflected, refracted);
    // // }
    //
    // bool Surface::timeToIntersect(const Ray& r, double& t) const {
    //     // Note t should be a good guess coming in for stability.
    //     // Algorithm is:
    //     // x,y,z <- ray.position(t)
    //     // sag <- surface.sag(x,y)
    //     // if z == sag
    //     //   return
    //     // normVec <- surface.normal(x,y)
    //     // plane <- Plane((x,y,sag), normVec)
    //     // t <- plane.intersect(r)
    //     // x,z,y <- ray.position(t)
    //     // sag <- surface.sag(x,y)
    //     // if z == sag
    //     //   return
    //     // ...
    //     Vector3d rayPoint = r.positionAtTime(t);
    //     double surfaceZ = sag(rayPoint[0], rayPoint[1]);
    //     size_t iter=0;
    //     double err = std::abs(surfaceZ - rayPoint[2]);
    //     while (err > 1e-14 && iter < 50) {
    //         Vector3d normVec = normal(rayPoint[0], rayPoint[1]);
    //         Vector3d surfacePoint{rayPoint[0], rayPoint[1], surfaceZ};
    //         t = normVec.dot(surfacePoint - r.r) / normVec.dot(r.v) + r.t;
    //         rayPoint = r.positionAtTime(t);
    //         surfaceZ = sag(rayPoint[0], rayPoint[1]);
    //         iter++;
    //         err = std::abs(surfaceZ - rayPoint[2]);
    //     }
    //     if (iter == 50)
    //         return false;
    //     return true;
    // }

}
