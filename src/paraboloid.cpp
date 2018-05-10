#include "paraboloid.h"
#include "utils.h"
#include <cmath>

namespace batoid {

    Paraboloid::Paraboloid(double R) : _R(R) {}

    std::string Paraboloid::repr() const {
        std::ostringstream oss(" ");
        oss << "Paraboloid(" << _R << ")";
        return oss.str();
    }

    double Paraboloid::sag(double x, double y) const {
        double r2 = x*x + y*y;
        if (_R != 0)
            return r2/(2*_R);
        return 0.0;
    }

    Vector3d Paraboloid::normal(double x, double y) const {
        if (_R == 0)
            return Vector3d(0,0,1);
        return Vector3d(-x/_R, -y/_R, 1).normalized();
    }

    bool Paraboloid::timeToIntersect(const Ray& r, double& t) const {
        double a = (r.v[0]*r.v[0] + r.v[1]*r.v[1])/2/_R;
        double b = (r.p0[0]*r.v[0] + r.p0[1]*r.v[1])/_R - r.v[2];
        double c = (r.p0[0]*r.p0[0] + r.p0[1]*r.p0[1])/2/_R - r.p0[2];
        double r1, r2;
        int n = solveQuadratic(a, b, c, r1, r2);

        if (n == 0)
            return false;
        else if (n == 1) {
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
        t += r.t0;
        return true;
    }

    Ray Paraboloid::intersect(const Ray& r) const {
        if (r.failed) return r;
        double t;
        if (!timeToIntersect(r, t))
            return Ray(true);
        Vector3d point = r.positionAtTime(t);
        return Ray(point, r.v, t, r.wavelength, r.isVignetted);
    }

    void Paraboloid::intersectInPlace(Ray& r) const {
        if (r.failed) return;
        double t;
        if (!timeToIntersect(r, t)) {
            r.failed=true;
            return;
        }
        r.p0 = r.positionAtTime(t);
        r.t0 = t;
        return;
    }

}
