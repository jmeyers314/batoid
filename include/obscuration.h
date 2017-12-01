#ifndef batoid_obscuration_h
#define batoid_obscuration_h

#include "vec2.h"
#include "ray.h"
#include <vector>
#include <memory>


namespace batoid {
    class Obscuration {
    public:
        virtual bool contains(double x, double y) const = 0;
        Ray obscure(const Ray&) const;
        void obscureInPlace(Ray&) const;
        std::vector<Ray> obscure(const std::vector<Ray>&) const;
        void obscureInPlace(std::vector<Ray>&) const;
    };

    class ObscCircle : public Obscuration {
    public:
        ObscCircle(double radius, double x0=0.0, double y0=0.0);
        bool contains(double x, double y) const override;
        const double _radius, _x0, _y0;
    };

    class ObscAnnulus : public Obscuration {
    public:
        ObscAnnulus(double inner, double outer, double x0=0.0, double y0=0.0);
        bool contains(double x, double y) const override;
        const double _inner, _outer, _x0, _y0;
    };

    class ObscRectangle : public Obscuration {
    public:
        ObscRectangle(double width, double height, double x0=0.0, double y0=0.0, double theta=0.0);
        bool contains(double x, double y) const override;
        const double _width, _height, _x0, _y0, _theta;
    private:
        Vec2 _A, _B, _C, _AB, _BC;
        double _ABAB, _BCBC;
    };

    class ObscRay : public Obscuration {
    public:
        ObscRay(double width, double theta, double x0=0.0, double y0=0.0);
        bool contains(double x, double y) const override;
        const double _width, _theta, _x0, _y0;
    private:
        const ObscRectangle _rect;
    };

    class ObscUnion : public Obscuration {
    public:
        ObscUnion(const std::vector<std::shared_ptr<Obscuration>> obscVec);
        bool contains(double x, double y) const override;
        const std::vector<std::shared_ptr<Obscuration>> _obscVec;
    };

    class ObscIntersection : public Obscuration {
    public:
        ObscIntersection(const std::vector<std::shared_ptr<Obscuration>> obscVec);
        bool contains(double x, double y) const override;
        const std::vector<std::shared_ptr<Obscuration>> _obscVec;
    };

    class ObscNegation : public Obscuration {
    public:
        ObscNegation(const std::shared_ptr<Obscuration> original);
        bool contains(double x, double y) const override;
        const std::shared_ptr<Obscuration> _original;
    };
}

#endif
