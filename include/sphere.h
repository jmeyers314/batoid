#ifndef batoid_sphere_h
#define batoid_sphere_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Sphere : public Surface {
    public:
        Sphere(double R);
        virtual double sag(double, double) const;
        virtual Vector3d normal(double, double) const;
        virtual Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        double getR() const { return _R; }
        std::string repr() const;

    private:
        const double _R;  // Radius of curvature

        bool timeToIntersect(const Ray& r, double& t) const;
        double dzdr(double r) const;
    };

    inline bool operator==(const Sphere& s1, const Sphere& s2)
        { return s1.getR() == s2.getR(); }
    inline bool operator!=(const Sphere& s1, const Sphere& s2)
        { return s1.getR() != s2.getR(); }
}
#endif
