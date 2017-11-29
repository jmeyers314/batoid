#include "batoid.h"
#include "ray.h"
#include "surface.h"
#include "medium.h"
#include "utils.h"
#include "intersection.h"
#include <numeric>

namespace batoid{
    Ray reflect(const Ray& r, const Surface& surface) {
        if (r.failed)
            return Ray(true);
        double n = 1.0 / r.v.Magnitude();
        Vec3 nv = r.v * n;
        Vec3 normal = surface.normal(r.p0.x, r.p0.y);
        double c1 = DotProduct(nv, normal);
        return Ray(r.p0, (nv - 2*c1*normal).UnitVec3()/n, r.t0, r.wavelength, r.isVignetted);
    }

    std::vector<Ray> reflect(const std::vector<Ray>& rays, const Surface& surface) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return reflect(r, surface); },
            2000
        );
        return result;
    }

    Ray refract(const Ray& r, const Surface& surface, const double n1, const double n2) {
        if (r.failed)
            return Ray(true);
        Vec3 nv = r.v * n1;
        Vec3 normal = surface.normal(r.p0.x, r.p0.y);
        double alpha = DotProduct(nv, normal);
        double a = 1.;
        double b = 2*alpha;
        double c = (1. - (n2*n2)/(n1*n1));
        double k1, k2;
        solveQuadratic(a, b, c, k1, k2);
        Vec3 f1 = (nv+k1*normal).UnitVec3();
        Vec3 f2 = (nv+k2*normal).UnitVec3();
        if (DotProduct(f1, nv) > DotProduct(f2, nv))
            return Ray(r.p0, f1/n2, r.t0, r.wavelength, r.isVignetted);
        else
            return Ray(r.p0, f2/n2, r.t0, r.wavelength, r.isVignetted);

    }

    std::vector<Ray> refract(const std::vector<Ray>& rays, const Surface& surface,
                             const double n1, const double n2) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return refract(r, surface, n1, n2); },
            2000
        );
        return result;
    }

    Ray refract(const Ray& r, const Surface& surface, const Medium& m1, const Medium& m2) {
        if (r.failed)
            return Ray(true);
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        return refract(r, surface, n1, n2);
    }

    std::vector<Ray> refract(const std::vector<Ray>& rays, const Surface& surface,
                             const Medium& m1, const Medium& m2) {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(
            rays.cbegin(), rays.cend(), result.begin(),
            [&](const Ray& r){ return refract(r, surface, m1, m2); },
            2000
        );
        return result;
    }

    std::vector<Ray> rayGrid(double dist, double length,
                             double xcos, double ycos,
                             int nside, double wavelength,
                             double n) {
        std::vector<Ray> result;
        result.reserve(nside*nside);

        // The "velocities" of all the rays in the grid are the same.
        auto v = Vec3(xcos, ycos, -sqrt(1-xcos*xcos-ycos*ycos))/n;

        double dx = length/(nside-1);
        double x0 = -length/2;
        double x = x0;
        for(int ix=0; ix<nside; ix++) {
            double y = x0;
            for(int iy=0; iy<nside; iy++) {
                Vec3 r(x,y,0);  // The position of the ray when it intersects the pupil
                // We want adjust the t0 of the Rays such that for a given time t, they all
                // lie on a plane perpendicular to v.  Can do this by solving
                // DotProduct(r + v*t + n*v*dist, v) == 0 for t
                // implies
                // t = -r.v / v.v - n*d
                //   = -r.v * n^2 - n*d
                double t = -DotProduct(r,v)*n*n - n*dist;
                result.push_back(Ray(r,v,-t,wavelength,false).propagatedToTime(0));
                y += dx;
            }
            x += dx;
        }
        return result;
    }

    std::vector<Ray> rayGrid(double dist, double length,
                             double xcos, double ycos,
                             int nside, double wavelength,
                             const Medium& m) {
        double n = m.getN(wavelength);
        return rayGrid(dist, length, xcos, ycos, nside, wavelength, n);
    }

    std::vector<Ray> circularGrid(double dist, double outer, double inner,
                                  double xcos, double ycos,
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
        auto v = Vec3(xcos, ycos, -sqrt(1-xcos*xcos-ycos*ycos))/n;

        rfrac = 1.0;
        for (int i=0; i<nradii; i++) {
            double az = 0.0;
            double daz = 2*M_PI/nphis[i];
            double radius = rfrac*outer;
            for (int j=0; j<nphis[i]; j++) {
                Vec3 r(radius*std::cos(az), radius*std::sin(az), 0);
                double t = -DotProduct(r,v)*n*n - n*dist;
                result.push_back(Ray(r,v,-t,wavelength,false).propagatedToTime(0));
                az += daz;
            }
            rfrac -= drfrac;
        }
        return result;
    }

    std::vector<Ray> circularGrid(double dist, double outer, double inner,
                                  double xcos, double ycos,
                                  int nradii, int naz, double wavelength, const Medium& m) {
        double n = m.getN(wavelength);
        return circularGrid(dist, outer, inner, xcos, ycos, nradii, naz, wavelength, n);
    }

    std::vector<Ray> trimVignetted(const std::vector<Ray>& rays) {
        std::vector<Ray> result;
        result.reserve(rays.size());
        for (const auto& r : rays) {
            if (!r.isVignetted)
                result.push_back(r);
        }
        return result;
    }

}
