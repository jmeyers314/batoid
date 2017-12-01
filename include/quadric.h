#ifndef batoid_quadric_h
#define batoid_quadric_h

#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {

    class Quadric : public Surface {
    public:
        Quadric(double R, double conic);
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        Ray intercept(const Ray&) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        virtual void interceptInPlace(Ray&) const;

        double getR() const {return _R;}
        double getConic() const {return _conic;}
        std::string repr() const;


    protected:
        bool timeToIntercept(const Ray& r, double& t) const;
        double dzdr(double r) const;

    private:
        const double _R;  // Radius of curvature
        const double _conic;  // Conic constant

    };

    inline std::ostream& operator<<(std::ostream& os, const Quadric& q);
    inline bool operator==(const Quadric& q1, const Quadric& q2)
        { return q1.getR() == q2.getR() && q1.getConic() == q2.getConic(); }
    inline bool operator!=(const Quadric& q1, const Quadric& q2)
        { return q1.getR() != q2.getR() || q1.getConic() != q2.getConic(); }

}
#endif
