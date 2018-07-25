#include "sphere.h"
#include "utils.h"

namespace batoid {
    Sphere::Sphere(double R) : _R(R), _Rsq(R*R), _Rinv(1./R), _Rinvsq(1./R/R) {}

    double Sphere::sag(double x, double y) const {
        if (_R != 0)
            return _R*(1-std::sqrt(1-(x*x + y*y)*_Rinvsq));
        return 0.0;
    }

    Vector3d Sphere::normal(double x, double y) const {
        if (_R == 0.0)
            return Vector3d(0,0,1);
        double r = std::sqrt(x*x + y*y);
        if (r == 0)
            return Vector3d(0,0,1);
        double dzdr1 = dzdr(r);
        return Vector3d(-dzdr1*x/r, -dzdr1*y/r, 1).normalized();
    }

    bool Sphere::timeToIntersect(const Ray& r, double& t) const {
        double vr2 = r.v[0]*r.v[0] + r.v[1]*r.v[1];
        double vz2 = r.v[2]*r.v[2];
        double vrr0 = r.v[0]*r.r[0] + r.v[1]*r.r[1];
        double r02 = r.r[0]*r.r[0] + r.r[1]*r.r[1];
        double z0term = (r.r[2]-_R);

        // Quadratic equation coefficients
        double a = vz2 + vr2;
        double b = 2*r.v[2]*z0term + 2*vrr0;
        double c = z0term*z0term - _Rsq + r02;

        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

        // Should probably check the solutions here since we obtained the quadratic
        // formula above by squaring both sides of an equation.
        if (n == 0) {
            return false;
        } else if (n == 1) {
            if (r1 < 0)
                return false;
            t = r1;
        } else {
            if (r1 < 0) {
                if (r2 < 0)
                    return false;
                else
                    t = r2;
            } else
                t = std::min(r1, r2);
        }
        t += r.t;
        return true;
    }

    bool Sphere::operator==(const Surface& rhs) const {
        if (const Sphere* other = dynamic_cast<const Sphere*>(&rhs)) {
            return _R == other->_R;
        } else return false;
    }

    double Sphere::dzdr(double r) const {
        double rat = r*_Rinv;
        return rat/std::sqrt(1-rat*rat);
    }

}
