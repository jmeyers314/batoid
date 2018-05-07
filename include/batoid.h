#ifndef batoid_batoid_h
#define batoid_batoid_h

#include "ray.h"
#include "rayVector.h"
#include "medium.h"
#include "surface.h"
#include "utils.h"

namespace batoid {
    Ray refract(const Ray& r, const Surface& surface, const Medium& m1, const Medium& m2);
    RayVector refract(const RayVector& rv, const Surface& surface, const Medium& m1, const Medium& m2);
    Ray refract(const Ray& r, const Surface& surface, const double n1, const double n2);

    Ray reflect(const Ray& r, const Surface& surface);
    RayVector reflect(const RayVector& rv, const Surface& surface);

    void refractInPlace(Ray& r, const Surface& surface, const Medium& m1, const Medium& m2);
    void refractInPlace(RayVector& rv, const Surface& surface, const Medium& m1, const Medium& m2);
    void refractInPlace(Ray& r, const Surface& surface, double n1, double n2);

    void reflectInPlace(Ray& r, const Surface& surface);
    void reflectInPlace(RayVector& rv, const Surface& surface);

    RayVector rayGrid(double dist, double length, double xcos, double ycos, double zcos,
                      int nside, double wavelength, const Medium& m);
    RayVector circularGrid(double dist, double outer, double inner,
                           double xcos, double ycos, double zcos,
                           int nradii, int naz, double wavelength, const Medium& m);
}

#endif
