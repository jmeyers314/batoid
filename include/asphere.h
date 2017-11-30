#ifndef batoid_asphere_h
#define batoid_asphere_h

#include <vector>
#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"
#include "quadric.h"

namespace batoid {

    class Asphere : public Quadric {
    public:
        Asphere(double R, double conic, std::vector<double> coefs);
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Ray intercept(const Ray&) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        virtual void interceptInPlace(Ray&) const;

        const std::vector<double>& getCoefs() const { return _coefs; }
        std::string repr() const;

    private:
        const std::vector<double> _coefs;  // Aspheric even polynomial coefficients

        bool timeToIntercept(const Ray& r, double& t) const;
        double dzdr(double r) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Asphere& a);

}
#endif
