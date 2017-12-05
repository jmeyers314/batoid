#ifndef batoid_paraboloid_h
#define batoid_paraboloid_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {

    class Paraboloid : public Surface {
    public:
        Paraboloid(double R);
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        virtual Ray intercept(const Ray&) const;
        virtual void interceptInPlace(Ray&) const;

        double getR() const {return _R;}
        std::string repr() const;

    private:
        const double _R;  // Radius of curvature

        bool timeToIntercept(const Ray& r, double& t) const;
    };

    inline bool operator==(const Paraboloid& p1, const Paraboloid& p2)
        { return p1.getR() == p2.getR(); }
    inline bool operator!=(const Paraboloid& p1, const Paraboloid& p2)
        { return p1.getR() != p2.getR(); }

}
#endif
