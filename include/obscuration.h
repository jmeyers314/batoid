#ifndef batoid_obscuration_h
#define batoid_obscuration_h

#include "vec2.h"
#include <vector>
#include <memory>

namespace batoid {
    class Obscuration {
    public:
        virtual bool contains(double x, double y) const = 0;
    };

    class ObscCircle : public Obscuration {
    public:
        ObscCircle(double x0, double y0, double radius);
        bool contains(double, double) const override;
    private:
        const double x0, y0, radius;
    };

    class ObscRectangle : public Obscuration {
    public:
        ObscRectangle(double x0, double y0, double width, double height, double theta);
        bool contains(double, double) const override;
    private:
        const double x0, y0, width, height, theta;
        Vec2 A, B, C, AB, BC;
        double ABAB, BCBC;
    };

    class ObscRay : public Obscuration {
    public:
        ObscRay(double x0, double y0, double width, double theta);
        bool contains(double, double) const override;
    private:
        const double x0, y0, width, theta;
        const ObscRectangle rect;
    };

    class ObscUnion : public Obscuration {
    public:
        ObscUnion(const std::vector<std::shared_ptr<Obscuration>> _obscVec);
        bool contains(double, double) const override;
    private:
        const std::vector<std::shared_ptr<Obscuration>> obscVec;
    };

    class ObscIntersection : public Obscuration {
    public:
        ObscIntersection(const std::vector<std::shared_ptr<Obscuration>> _obscVec);
        bool contains(double, double) const override;
    private:
        const std::vector<std::shared_ptr<Obscuration>> obscVec;
    };

    class ObscNegation : public Obscuration {
    public:
        ObscNegation(const std::shared_ptr<Obscuration> _original);
        bool contains(double, double) const override;
    private:
        const std::shared_ptr<Obscuration> original;
    };
}

#endif
