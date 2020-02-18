#ifndef batoid_batoid_h
#define batoid_batoid_h

#include "ray.h"
#include "rayVector.h"
#include "medium.h"
#include "surface.h"
#include "utils.h"

namespace batoid {
    RayVector rayGrid(
            double dist, double length, double xcos, double ycos, double zcos,
            int nside, double wavelength, double flux, const Medium& m,
            const CoordSys& coordSys, bool lattice);
    RayVector circularGrid(
            double dist, double outer, double inner,
            double xcos, double ycos, double zcos,
            int nradii, int naz, double wavelength, double flux, const Medium& m,
            const CoordSys& coordSys);
    RayVector uniformCircularGrid(
            double dist, double outer, double inner,
            double xcos, double ycos, double zcos,
            int nrays, double wavelength, double flux, const Medium& m, const CoordSys& coordSys,
            int seed);
    RayVector pointSourceCircularGrid(
            const Vector3d& source, double outer, double inner,
            int nradii, int naz, double wavelength, double flux, const Medium& m,
            const CoordSys& coordSys);
}

#endif
