#ifndef batoid_sphere_h
#define batoid_sphere_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {

    class Sphere : public Surface {
    public:
        Sphere(double R);
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Ray intercept(const Ray&) const;
        virtual void interceptInPlace(Ray&) const;

        double getR() const { return _R; }
        std::string repr() const;

    private:
        const double _R;  // Radius of curvature

        bool timeToIntercept(const Ray& r, double& t) const;
        double dzdr(double r) const;
    };

    inline bool operator==(const Sphere& s1, const Sphere& s2)
        { return s1.getR() == s2.getR(); }
    inline bool operator!=(const Sphere& s1, const Sphere& s2)
        { return s1.getR() != s2.getR(); }
}
#endif
