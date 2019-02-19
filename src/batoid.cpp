#include "batoid.h"
#include "ray.h"
#include "surface.h"
#include "medium.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid{
    RayVector rayGrid(double dist, double length,
                      double xcos, double ycos, double zcos,
                      int nside, double wavelength, double flux,
                      const Medium& m) {
        double n = m.getN(wavelength);
    // `dist` is the distance from the center of the pupil to the center of the rayGrid.
    // `length` is the length of one side of the rayGrid square.
    // `xcos`, `ycos`, `zcos` are the direction cosines of the ray velocities
    // `nside` is the number of rays on a side of the rayGrid.
    // `wavelength` is the wavelength assigned to the rays
    // `m` is the medium (from which we get the refractive index) at the position of the rays.
    // (Needed to properly normalize the ray magnitudes).
        std::vector<Ray> result;
        result.reserve(nside*nside);

        // The "velocities" of all the rays in the grid are the same.
        Vector3d v(xcos, ycos, zcos);
        v.normalize();
        v /= n;

        double dy = length/(nside-1);
        double y0 = -length/2;
        double y = y0;
        for(int iy=0; iy<nside; iy++) {
            double x = y0;
            for(int ix=0; ix<nside; ix++) {
                // Start with the position of the ray when it intersects the pupil
                Vector3d r(x,y,0);
                // We know that the position of the ray that goes through the origin
                // (which is also the center of the pupil), is given by
                //   a = -dist * vhat = -dist * v * n
                // We want to find the position r0 that satisfies
                // 1) r0 - a is perpendicular to v
                // 2) r = r0 + v t
                // The first equation can be rewritten as
                // (r0 - a) . v = 0
                // some algebra reveals
                // (r + v n d) . v - t v . v = 0
                // => t = (r + v n d) . v / v . v
                //      = (r + v n d) . v n^2
                // => r0 = r - v t
                double t = (r + v*n*dist).dot(v) * n * n;
                result.emplace_back(r-v*t, v, 0, wavelength, flux, false);
                x += dy;
            }
            y += dy;
        }
        return RayVector(std::move(result), wavelength);
    }

    RayVector circularGrid(double dist, double outer, double inner,
                           double xcos, double ycos, double zcos,
                           int nradii, int naz, double wavelength, double flux, const Medium& m) {
        double n = m.getN(wavelength);

        // Determine number of rays at each radius
        std::vector<int> nphis(nradii);
        double drfrac = (outer-inner)/(nradii-1)/outer;
        double rfrac = 1.0;
        for (int i=0; i<nradii; i++) {
            nphis[i] = int(std::ceil(naz*rfrac/6.))*6;
            rfrac -= drfrac;
        }
        // Point in the center is a special case
        if (inner == 0.0)
            nphis[nradii-1] = 1;
        int nray = std::accumulate(nphis.begin(), nphis.end(), 0);

        std::vector<Ray> result;
        result.reserve(nray);

        // The "velocities" of all the rays in the grid are the same.
        Vector3d v(xcos, ycos, zcos);
        v.normalize();
        v /= n;

        rfrac = 1.0;
        for (int i=0; i<nradii; i++) {
            double az = 0.0;
            double daz = 2*M_PI/nphis[i];
            double radius = rfrac*outer;
            for (int j=0; j<nphis[i]; j++) {
                Vector3d r(radius*std::cos(az), radius*std::sin(az), 0);
                double t = (r + v*n*dist).dot(v) * n * n;
                result.emplace_back(r-v*t, v, 0, wavelength, flux, false);
                az += daz;
            }
            rfrac -= drfrac;
        }
        return RayVector(std::move(result), wavelength);
    }
}
