#include "vec2.h"
#include "obscuration.h"
#include <cmath>

namespace batoid {
    ObscCircle::ObscCircle(double _x0, double _y0, double _radius) :
        x0(_x0), y0(_y0), radius(_radius) {}

    bool ObscCircle::contains(double x, double y) const {
        return std::hypot(x-x0, y-y0) < radius;
    }

    ObscRectangle::ObscRectangle(double _x0, double _y0, double w, double h, double th) :
        x0(_x0), y0(_y0), width(w), height(h), theta(th)
    {
        A = {-width/2, -height/2};
        B = {-width/2, +height/2};
        C = {+width/2, +height/2};
        double sth = std::sin(th);
        double cth = std::cos(th);
        Rot2 R{{{cth, -sth, sth, cth}}};
        Vec2 center(x0, y0);
        A = RotVec(R, A) + center;
        B = RotVec(R, B) + center;
        C = RotVec(R, C) + center;
        AB = B - A;
        ABAB = DotProduct(AB, AB);
        BC = C - B;
        BCBC = DotProduct(BC, BC);
    }

    bool ObscRectangle::contains(double x, double y) const {
        Vec2 M(x, y);
        Vec2 AM(M - A);
        Vec2 BM(M - B);
        double ABAM(DotProduct(AM, AB));
        double BCBM(DotProduct(BM, BC));
        return (0 <= ABAM) && (ABAM <= ABAB) && (0 <= BCBM) && (BCBM <= BCBC);
    }

    ObscRay::ObscRay(double _x0, double _y0, double w, double th) :
        x0(_x0), y0(_y0), width(w), theta(th),
        rect(ObscRectangle(x0 + 100*std::cos(th)/2,
                           y0 + 100*std::sin(th)/2,
                           100, width, theta)) {}

    bool ObscRay::contains(double x, double y) const {
        return rect.contains(x, y);
    }

    ObscUnion::ObscUnion(const std::vector<std::shared_ptr<Obscuration>> _obscVec) :
        obscVec(_obscVec) {}

    bool ObscUnion::contains(double x, double y) const {
        bool ret = false;
        for(const auto& obscPtr : obscVec) {
            ret = ret || obscPtr->contains(x, y);
        }
        return ret;
    }

    ObscIntersection::ObscIntersection(const std::vector<std::shared_ptr<Obscuration>> _obscVec) :
        obscVec(_obscVec) {}

    bool ObscIntersection::contains(double x, double y) const {
        bool ret = true;
        for(const auto& obscPtr : obscVec) {
            ret = ret && obscPtr->contains(x, y);
        }
        return ret;
    }

    ObscNegation::ObscNegation(const std::shared_ptr<Obscuration> _original) :
        original(_original) {}

    bool ObscNegation::contains(double x, double y) const {
        return not original->contains(x, y);
    }
}
