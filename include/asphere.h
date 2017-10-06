#ifndef batoid_asphere_h
#define batoid_asphere_h

#include <vector>
#include <sstream>
#include <limits>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {

    class Asphere : public Surface {
    public:
        Asphere(double _R, double _kappa, std::vector<double> _alpha, double _B,
                double _Rin=0.0, double _Rout=std::numeric_limits<double>::infinity());
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Ray intercept(const Ray&) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        double getR() const {return R;}
        double getKappa() const {return kappa;}
        const std::vector<double>& getAlpha() const {return alpha;}
        double getB() const {return B;}
        double getRin() const {return Rin;}
        double getRout() const {return Rout;}

        std::string repr() const;

    private:
        const double R, kappa;
        const std::vector<double> alpha;
        const double B;
        const double Rin, Rout;  // Inner and outer radii for vignetting

        double dzdr(double r) const;
    };

    inline std::ostream& operator<<(std::ostream& os, const Asphere& a);

}
#endif
