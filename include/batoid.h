#ifndef batoid_batoid_h
#define batoid_batoid_h

#include "ray.h"
#include "medium.h"
#include "surface.h"
#include "utils.h"

namespace batoid {
    Ray refract(const Ray& r, const Surface& surface, const Medium& m1, const Medium& m2);
    std::vector<Ray> refract(const std::vector<Ray>& r, const Surface& surface, const Medium& m1, const Medium& m2);
    Ray refract(const Ray& r, const Surface& surface, const double n1, const double n2);
    std::vector<Ray> refract(const std::vector<Ray>& r, const Surface& surface, const double n1, const double n2);

    Ray reflect(const Ray& r, const Surface& surface);
    std::vector<Ray> reflect(const std::vector<Ray>& r, const Surface& surface);

    void refractInPlace(Ray& r, const Surface& surface, const Medium& m1, const Medium& m2);
    void refractInPlace(std::vector<Ray>& rays, const Surface& surface, const Medium& m1, const Medium& m2);
    void refractInPlace(Ray& r, const Surface& surface, double n1, double n2);
    void refractInPlace(std::vector<Ray>& rays, const Surface& surface, double n1, double n2);

    void reflectInPlace(Ray& r, const Surface& surface);
    void reflectInPlace(std::vector<Ray>& r, const Surface& surface);

    std::vector<Ray> rayGrid(double dist, double length, double xcos, double ycos, double zcos,
                             int nside, double wavelength, double n);
    std::vector<Ray> rayGrid(double dist, double length, double xcos, double ycos, double zcos,
                             int nside, double wavelength, const Medium& m);

    std::vector<Ray> circularGrid(double dist, double outer, double inner,
                                  double xcos, double ycos, double zcos,
                                  int nradii, int naz, double wavelength, double n);
    std::vector<Ray> circularGrid(double dist, double outer, double inner,
                                  double xcos, double ycos, double zcos,
                                  int nradii, int naz, double wavelength, const Medium& m);

    std::vector<Ray> trimVignetted(const std::vector<Ray>& rays);
    void trimVignettedInPlace(std::vector<Ray>& rays);
}

#endif
