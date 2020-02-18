#include "coordtransform2.h"
#include "utils.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

using Eigen::Vector3d;
using Eigen::Matrix3d;

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

    CoordTransform2::CoordTransform2(const CoordSys& source, const CoordSys& destination) :
        _dr(source.m_rot.transpose()*(destination.m_origin - source.m_origin)),
        _rot(source.m_rot.transpose()*destination.m_rot),
        _source(source), _destination(destination) {}

    CoordTransform2::CoordTransform2(const Vector3d& dr, const Matrix3d& rot) :
        _dr(dr), _rot(rot) {}

    void CoordTransform2::applyForwardInPlace(RayVector2& rv) const {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.deviceData;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        const double* rot = _rot.data();
        const double* dr = _dr.data();
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr) map(to:rot[:9],dr[:3])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double x = (xptr[i]-dr[0])*rot[0] + (yptr[i]-dr[1])*rot[1] + (zptr[i]-dr[2])*rot[2];
                double y = (xptr[i]-dr[0])*rot[3] + (yptr[i]-dr[1])*rot[4] + (zptr[i]-dr[2])*rot[5];
                double z = (xptr[i]-dr[0])*rot[6] + (yptr[i]-dr[1])*rot[7] + (zptr[i]-dr[2])*rot[8];
                double vx = vxptr[i]*rot[0] + vyptr[i]*rot[1] + vzptr[i]*rot[2];
                double vy = vxptr[i]*rot[3] + vyptr[i]*rot[4] + vzptr[i]*rot[5];
                double vz = vxptr[i]*rot[6] + vyptr[i]*rot[7] + vzptr[i]*rot[8];
                xptr[i] = x;
                yptr[i] = y;
                zptr[i] = z;
                vxptr[i] = vx;
                vyptr[i] = vy;
                vzptr[i] = vz;
            }
        }
    }

    void CoordTransform2::applyReverseInPlace(RayVector2& rv) const {
        rv.r.syncToDevice();
        rv.v.syncToDevice();
        size_t size = rv.size;
        double* xptr = rv.r.deviceData;
        double* yptr = xptr + size;
        double* zptr = yptr + size;
        double* vxptr = rv.v.deviceData;
        double* vyptr = vxptr + size;
        double* vzptr = vyptr + size;
        const double* rot = _rot.data();
        const double* dr = _dr.data();
        #pragma omp target is_device_ptr(xptr, yptr, zptr, vxptr, vyptr, vzptr) map(to:rot[:9],dr[:3])
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++) {
                double x = xptr[i]*rot[0] + yptr[i]*rot[3] + zptr[i]*rot[6] + dr[0];
                double y = xptr[i]*rot[1] + yptr[i]*rot[4] + zptr[i]*rot[7] + dr[1];
                double z = xptr[i]*rot[2] + yptr[i]*rot[5] + zptr[i]*rot[8] + dr[2];
                double vx = vxptr[i]*rot[0] + vyptr[i]*rot[3] + vzptr[i]*rot[6];
                double vy = vxptr[i]*rot[1] + vyptr[i]*rot[4] + vzptr[i]*rot[7];
                double vz = vxptr[i]*rot[2] + vyptr[i]*rot[5] + vzptr[i]*rot[8];
                xptr[i] = x;
                yptr[i] = y;
                zptr[i] = z;
                vxptr[i] = vx;
                vyptr[i] = vy;
                vzptr[i] = vz;
            }
        }
    }
}
