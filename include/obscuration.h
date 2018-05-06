#ifndef batoid_obscuration_h
#define batoid_obscuration_h

#include "ray.h"
#include "rayVector.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>

using Eigen::Vector2d;
using Eigen::Matrix2d;

namespace batoid {
    class Obscuration {
    public:
        virtual bool contains(double x, double y) const = 0;
        virtual bool operator==(const Obscuration& rhs) const = 0;
        bool operator!=(const Obscuration& rhs) const { return !operator==(rhs); }
        Ray obscure(const Ray&) const;
        void obscureInPlace(Ray&) const;
        std::vector<Ray> obscure(const std::vector<Ray>&) const;
        void obscureInPlace(std::vector<Ray>&) const;
        virtual std::string repr() const = 0;
    };
    std::ostream& operator<<(std::ostream& os, const Obscuration& o);

    class ObscCircle : public Obscuration {
    public:
        ObscCircle(double radius, double x0=0.0, double y0=0.0);
        bool contains(double x, double y) const override;
        bool operator==(const Obscuration& rhs) const override;
        std::string repr() const override;

        const double _radius, _x0, _y0;
    };

    class ObscAnnulus : public Obscuration {
    public:
        ObscAnnulus(double inner, double outer, double x0=0.0, double y0=0.0);
        bool contains(double x, double y) const override;
        bool operator==(const Obscuration& rhs) const override;
        std::string repr() const override;

        const double _inner, _outer, _x0, _y0;
    };

    class ObscRectangle : public Obscuration {
    public:
        ObscRectangle(double width, double height, double x0=0.0, double y0=0.0, double theta=0.0);
        bool contains(double x, double y) const override;
        bool operator==(const Obscuration& rhs) const override;
        std::string repr() const override;

        const double _width, _height, _x0, _y0, _theta;
    private:
        Vector2d _A, _B, _C, _AB, _BC;
        double _ABAB, _BCBC;
    };

    class ObscRay : public Obscuration {
    public:
        ObscRay(double width, double theta, double x0=0.0, double y0=0.0);
        bool contains(double x, double y) const override;
        bool operator==(const Obscuration& rhs) const override;
        std::string repr() const override;

        const double _width, _theta, _x0, _y0;
    private:
        const ObscRectangle _rect;
    };

    class ObscNegation : public Obscuration {
    public:
        ObscNegation(const std::shared_ptr<Obscuration> original);
        bool contains(double x, double y) const override;
        bool operator==(const Obscuration& rhs) const override;
        std::string repr() const override;

        const std::shared_ptr<Obscuration> _original;
    };

    class ObscUnion : public Obscuration {
    public:
        ObscUnion(const std::vector<std::shared_ptr<Obscuration>> obscVec);
        bool contains(double x, double y) const override;
        bool operator==(const Obscuration& rhs) const override;
        std::string repr() const override;

        // Don't make const, so can sort in the ctor
        std::vector<std::shared_ptr<Obscuration>> _obscVec;
    };

    class ObscIntersection : public Obscuration {
    public:
        ObscIntersection(const std::vector<std::shared_ptr<Obscuration>> obscVec);
        bool contains(double x, double y) const override;
        bool operator==(const Obscuration& rhs) const override;
        std::string repr() const override;

        // Don't make const, so can sort in the ctor
        std::vector<std::shared_ptr<Obscuration>> _obscVec;
    };
}

#endif
