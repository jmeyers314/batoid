#ifndef batoid_batoid_h
#define batoid_batoid_h

#include "vec3.h"
#include "ray.h"
#include "intersection.h"
#include "surface.h"
#include "utils.h"

namespace batoid {
    // Ray intercept(const Ray& r, const Surface& surface);
    // std::vector<Ray> intercept(const std::vector<Ray>& r, const Surface& surface);

    Ray refract(const Ray& r, const Surface& surface, const Medium& m1, const Medium& m2);
    std::vector<Ray> refract(const std::vector<Ray>& r, const Surface& surface, const Medium& m1, const Medium& m2);
    Ray refract(const Ray& r, const Surface& surface, const double n1, const double n2);
    std::vector<Ray> refract(const std::vector<Ray>& r, const Surface& surface, const double n1, const double n2);

    Ray reflect(const Ray& r, const Surface& surface);
    std::vector<Ray> reflect(const std::vector<Ray>& r, const Surface& surface);
}

#endif
