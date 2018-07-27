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

        double t1, t2;
        int n = solveQuadratic(a, b, c, t1, t2);

        // Should probably check the solutions here since we obtained the quadratic
        // formula above by squaring both sides of an equation.
        if (n == 0) {
            return false;
        } else if (n == 1) {
            if (t1 < 0)
                return false;
            t = t1;
        } else {
            if (t1 < 0) {
                if (t2 < 0)
                    return false;
                else
                    t = t2;
            } else {
                // We need to pick whichever time is most consistent with the sag.
                Ray r1 = r.propagatedToTime(r.t + t1);
                Ray r2 = r.propagatedToTime(r.t + t2);
                double d1 = std::abs(sag(r1.r[0], r1.r[1]) - r1.r[2]);
                double d2 = std::abs(sag(r2.r[0], r2.r[1]) - r2.r[2]);
                t = (d1 < d2) ? t1 : t2;
            }
        }
        t += r.t;
        return true;
    }

    double Sphere::dzdr(double r) const {
        double rat = r*_Rinv;
        return rat/std::sqrt(1-rat*rat);
    }

}
