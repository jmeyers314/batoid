#include "coordTransform.h"

namespace batoid {

    // x is global coordinate
    // y is destination, with corresponding R and dr
    // z is source, with corresponding S and ds
    //
    // y = Rinv(x-dr)
    // z = Sinv(x-ds)
    // implies
    // x = S z + ds
    //
    // y = Rinv(S z + ds - dr)
    //   = Rinv S z + Rinv ds - Rinv dr
    //   = Rinv S z + Rinv S Sinv ds - Rinv S Sinv dr
    //   = Rinv S (z + Sinv ds - Sinv dr)
    //   = (Sinv R)^-1 (z - (Sinv dr - Sinv ds))
    //   = (Sinv R)^-1 (z - Sinv (dr - ds))
    //   = Tinv^-1 (z - dt)
    //   where T = Sinv R
    //   and  dt = Sinv (dr - ds)
    CoordTransform::CoordTransform(const CoordSys& _source, const CoordSys& _destination) :
        dr(transpose(_source.m_rot)*(_destination.m_origin - _source.m_origin)),
        rot(transpose(_source.m_rot)*_destination.m_rot),
        source(_source),
        destination(_destination)
    {}

    // We actively shift and rotate the coordinate system axes,
    // This looks like y = R x + dr
    // For a passive transformation of a fixed vector from one coord sys to another
    // though, we want the opposite transformation: y = R^-1 (x - dr)
    void CoordTransform::applyForwardInPlace(RayVector& rv) const {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = xptr + 2*size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vxptr + 2*size;
        const double* drptr = dr.data();
        const double* rotptr = rot.data();

        #pragma omp target teams distribute parallel for map(to: drptr[:3], rotptr[:9])
        for(int i=0; i<size; i++) {
            double dx = xptr[i]-drptr[0];
            double dy = yptr[i]-drptr[1];
            double dz = zptr[i]-drptr[2];
            xptr[i] = dx*rotptr[0] + dy*rotptr[3] + dz*rotptr[6];
            yptr[i] = dx*rotptr[1] + dy*rotptr[4] + dz*rotptr[7];
            zptr[i] = dx*rotptr[2] + dy*rotptr[5] + dz*rotptr[8];
            double vx = vxptr[i]*rotptr[0] + vyptr[i]*rotptr[3] + vzptr[i]*rotptr[6];
            double vy = vxptr[i]*rotptr[1] + vyptr[i]*rotptr[4] + vzptr[i]*rotptr[7];
            double vz = vxptr[i]*rotptr[2] + vyptr[i]*rotptr[5] + vzptr[i]*rotptr[8];
            vxptr[i] = vx;
            vyptr[i] = vy;
            vzptr[i] = vz;
        }
    }

    void CoordTransform::applyReverseInPlace(RayVector& rv) const {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.data;
        double* yptr = xptr + size;
        double* zptr = xptr + 2*size;
        double* vxptr = rv.v.data;
        double* vyptr = vxptr + size;
        double* vzptr = vxptr + 2*size;
        const double* drptr = dr.data();
        const double* rotptr = rot.data();

        #pragma omp target teams distribute parallel for map(to: drptr[:3], rotptr[:9])
        for(int i=0; i<size; i++) {
            double x = xptr[i]*rotptr[0] + yptr[i]*rotptr[1] + zptr[i]*rotptr[2] + drptr[0];
            double y = xptr[i]*rotptr[3] + yptr[i]*rotptr[4] + zptr[i]*rotptr[5] + drptr[1];
            double z = xptr[i]*rotptr[6] + yptr[i]*rotptr[7] + zptr[i]*rotptr[8] + drptr[2];
            xptr[i] = x;
            yptr[i] = y;
            zptr[i] = z;
            double vx = vxptr[i]*rotptr[0] + vyptr[i]*rotptr[1] + vzptr[i]*rotptr[2];
            double vy = vxptr[i]*rotptr[3] + vyptr[i]*rotptr[4] + vzptr[i]*rotptr[5];
            double vz = vxptr[i]*rotptr[6] + vyptr[i]*rotptr[7] + vzptr[i]*rotptr[8];
            vxptr[i] = vx;
            vyptr[i] = vy;
            vzptr[i] = vz;
        }
    }
}
