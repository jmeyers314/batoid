#include "batoid.h"
#include "ray.h"
#include "surface.h"
#include "medium.h"
#include "utils.h"
#include <numeric>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid{
    Ray reflect(const Ray& r, const Surface& surface) {
        if (r.failed) return r;
        double n = 1.0 / r.v.norm();
        Vector3d nv = r.v * n;
        Vector3d normal(surface.normal(r.p0[0], r.p0[1]));
        double c1 = nv.dot(normal);
        return Ray(r.p0, (nv - 2*c1*normal).normalized()/n, r.t0, r.wavelength, r.isVignetted);
    }

    std::vector<Ray> reflect(const std::vector<Ray>& rays, const Surface& surface) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return reflect(r, surface); }
        );
        return result;
    }

    void reflectInPlace(Ray& r, const Surface& surface) {
        if (r.failed) return;
        double n = 1.0 / r.v.norm();
        Vector3d nv = r.v * n;
        Vector3d normal(surface.normal(r.p0[0], r.p0[1]));
        double c1 = nv.dot(normal);
        r.v = (nv - 2*c1*normal).normalized()/n;
    }

    void reflectInPlace(std::vector<Ray>& rays, const Surface& surface) {
        parallel_for_each(
            rays.begin(), rays.end(),
            [&](Ray& r) { reflectInPlace(r, surface); }
        );
    }

    Ray refract(const Ray& r, const Surface& surface, const double n1, const double n2) {
        if (r.failed) return r;
        Vector3d nv = r.v * n1;
        Vector3d normal(surface.normal(r.p0[0], r.p0[1]));
        double alpha = nv.dot(normal);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vector3d f1 = (nv+k1*normal).normalized();
        Vector3d f2 = (nv+k2*normal).normalized();
        if (f1.dot(nv) > f2.dot(nv))
            return Ray(r.p0, f1/n2, r.t0, r.wavelength, r.isVignetted);
        else
            return Ray(r.p0, f2/n2, r.t0, r.wavelength, r.isVignetted);
    }

    std::vector<Ray> refract(const std::vector<Ray>& rays, const Surface& surface,
                             const double n1, const double n2) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return refract(r, surface, n1, n2); }
        );
        return result;
    }

    Ray refract(const Ray& r, const Surface& surface, const Medium& m1, const Medium& m2) {
        if (r.failed) return r;
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        return refract(r, surface, n1, n2);
    }

    std::vector<Ray> refract(const std::vector<Ray>& rays, const Surface& surface,
                             const Medium& m1, const Medium& m2) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return refract(r, surface, m1, m2); }
        );
        return result;
    }

    void refractInPlace(Ray& r, const Surface& surface, double n1, double n2) {
        if (r.failed) return;
        Vector3d nv = r.v * n1;
        Vector3d normal(surface.normal(r.p0[0], r.p0[1]));
        double alpha = nv.dot(normal);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vector3d f1 = (nv+k1*normal).normalized();
        Vector3d f2 = (nv+k2*normal).normalized();
        if (f1.dot(nv) > f2.dot(nv))
            r.v = f1/n2;
        else
            r.v = f2/n2;
    }

    void refractInPlace(std::vector<Ray>& rays, const Surface& surface, double n1, double n2) {
        parallel_for_each(
            rays.begin(), rays.end(),
            [&](Ray& r) { refractInPlace(r, surface, n1, n2); }
        );
    }

    void refractInPlace(Ray& r, const Surface& surface, const Medium& m1, const Medium& m2) {
        if (r.failed) return;
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        refractInPlace(r, surface, n1, n2);
    }

    void refractInPlace(std::vector<Ray>& rays, const Surface& surface, const Medium& m1, const Medium& m2) {
        parallel_for_each(
            rays.begin(), rays.end(),
            [&](Ray& r) { refractInPlace(r, surface, m1, m2); }
        );
    }

    std::vector<Ray> rayGrid(double dist, double length,
                             double xcos, double ycos, double zcos,
                             int nside, double wavelength,
                             double n) {
    // `dist` is the distance from the center of the pupil to the center of the rayGrid.
    // `length` is the length of one side of the rayGrid square.
    // `xcos`, `ycos`, `zcos` are the direction cosines of the ray velocities
    // `nside` is the number of rays on a side of the rayGrid.
    // `wavelength` is the wavelength assigned to the rays
    // `n` is the refractive index at the position of the rays.  (Needed to properly normalize the
    //     ray magnitudes).
        std::vector<Ray> result;
        result.reserve(nside*nside);

        // The "velocities" of all the rays in the grid are the same.
        Vector3d v(xcos, ycos, zcos);
        v.normalize();
        v /= n;

        double dy = length/nside;
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
                result.push_back(Ray(r-v*t, v, 0, wavelength, false));
                x += dy;
            }
            y += dy;
        }
        return result;
    }

    std::vector<Ray> rayGrid(double dist, double length,
                             double xcos, double ycos, double zcos,
                             int nside, double wavelength,
                             const Medium& m) {
        double n = m.getN(wavelength);
        return rayGrid(dist, length, xcos, ycos, zcos, nside, wavelength, n);
    }

    std::vector<Ray> circularGrid(double dist, double outer, double inner,
                                  double xcos, double ycos, double zcos,
                                  int nradii, int naz, double wavelength, double n) {

        // Determine number of rays at each radius
        std::vector<int> nphis(nradii);
        double drfrac = (outer-inner)/(nradii-1)/outer;
        double rfrac = 1.0;
        for (int i=0; i<nradii; i++) {
            nphis[i] = int(std::ceil(naz*rfrac));
            rfrac -= drfrac;
        }
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
                result.push_back(Ray(r-v*t, v, 0, wavelength, false));
                az += daz;
            }
            rfrac -= drfrac;
        }
        return result;
    }

    std::vector<Ray> circularGrid(double dist, double outer, double inner,
                                  double xcos, double ycos, double zcos,
                                  int nradii, int naz, double wavelength, const Medium& m) {
        double n = m.getN(wavelength);
        return circularGrid(dist, outer, inner, xcos, ycos, zcos, nradii, naz, wavelength, n);
    }

    std::vector<Ray> trimVignetted(const std::vector<Ray>& rays) {
        std::vector<Ray> result;
        result.reserve(rays.size());
        std::copy_if(
            rays.begin(),
            rays.end(),
            std::back_inserter(result),
            [](const Ray& r){return !r.isVignetted;}
        );
        return result;
    }

    void trimVignettedInPlace(std::vector<Ray>& rays) {
        rays.erase(
            std::remove_if(
                rays.begin(),
                rays.end(),
                [](const Ray& r){ return r.failed || r.isVignetted; }
            ),
            rays.end()
        );
    }

}
