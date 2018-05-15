#ifndef batoid_asphere_h
#define batoid_asphere_h

#include <vector>
#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include "quadric.h"
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {

    class Asphere : public Quadric {
    public:
        Asphere(double R, double conic, std::vector<double> coefs);
        virtual double sag(double, double) const;
        virtual Vector3d normal(double, double) const;
        virtual Ray intersect(const Ray&) const;
        virtual void intersectInPlace(Ray&) const;

        const std::vector<double>& getCoefs() const { return _coefs; }
        std::string repr() const;

    private:
        const std::vector<double> _coefs;  // Aspheric even polynomial coefficients
        const std::vector<double> _dzdrcoefs;  // Coefficients for computing dzdr

        bool timeToIntersect(const Ray& r, double& t) const;
        double dzdr(double r) const;
        static std::vector<double> computeDzDrCoefs(const std::vector<double>& coefs);
    };

    inline bool operator==(const Asphere& a1, const Asphere& a2) {
        return a1.getR() == a2.getR() &&
        a1.getConic() == a2.getConic() &&
        a1.getCoefs() == a2.getCoefs();
    }
    inline bool operator!=(const Asphere& a1, const Asphere& a2) {
        return a1.getR() != a2.getR() ||
        a1.getConic() != a2.getConic() ||
        a1.getCoefs() != a2.getCoefs();
    }

}
#endif
