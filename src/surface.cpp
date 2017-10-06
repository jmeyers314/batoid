#include "surface.h"
#include "transformation.h"
#include "utils.h"

namespace batoid {
    Transformation Surface::shift(double dx, double dy, double dz) const {
        return Transformation(shared_from_this(), dx, dy, dz);
    }

    Transformation Surface::shift(const Vec3& dr) const {
        return Transformation(shared_from_this(), dr.x, dr.y, dr.z);
    }

    Transformation Surface::rotX(double theta) const {
        double st = std::sin(theta);
        double ct = std::cos(theta);
        Rot3 r{{{1,   0,  0,
                 0,  ct, st,
                 0, -st, ct}}};
        return Transformation(shared_from_this(), r);
    }

    Transformation Surface::rotY(double theta) const {
        double st = std::sin(theta);
        double ct = std::cos(theta);
        Rot3 r{{{ct, 0, -st,
                  0, 1,   0,
                 st, 0,  ct}}};
        return Transformation(shared_from_this(), r);
    }

    Transformation Surface::rotZ(double theta) const {
        double st = std::sin(theta);
        double ct = std::cos(theta);
        Rot3 r{{{ ct, st, 0,
                 -st, ct, 0,
                   0,  0, 1}}};
        return Transformation(shared_from_this(), r);
    }

    std::vector<Intersection> Surface::intersect(const std::vector<Ray>& rays) const {
        auto result = std::vector<Intersection>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [this](const Ray& ray)
            { return intersect(ray); },
            2000
        );
        return result;
    }

    std::vector<Ray> Surface::intercept(const std::vector<Ray>& rays) const {
        auto result = std::vector<Ray>(rays.size());
        parallelTransform(rays.cbegin(), rays.cend(), result.begin(),
            [this](const Ray& ray)
            { return intercept(ray); },
            2000
        );
        return result;
    }
}
