#ifndef batoid_obscuration2_h
#define batoid_obscuration2_h

#include "rayVector2.h"

namespace batoid {
    class Obscuration2 {
    public:
        Obscuration2() : _devPtr(nullptr) {}
        virtual Obscuration2* getDevPtr() const = 0;
        virtual ~Obscuration2() {}

        virtual bool contains(double x, double y) const = 0;

        void obscureInPlace(RayVector2&) const;
    protected:
        mutable Obscuration2* _devPtr;
    };


    class ObscCircle2 : public Obscuration2 {
    public:
        ObscCircle2(double radius, double x0=0.0, double y0=0.0);
        Obscuration2* getDevPtr() const override;
        bool contains(double x, double y) const override;

        const double _radius, _x0, _y0;
    };


    class ObscAnnulus2 : public Obscuration2 {
    public:
        ObscAnnulus2(double inner, double outer, double x0=0.0, double y0=0.0);
        Obscuration2* getDevPtr() const override;
        bool contains(double x, double y) const override;

        const double _inner, _outer, _x0, _y0;
    };


    class ObscRectangle2 : public Obscuration2 {
    public:
        ObscRectangle2(double width, double height, double x0=0.0, double y0=0.0, double theta=0.0);
        Obscuration2* getDevPtr() const override;
        bool contains(double x, double y) const override;

        const double _width, _height, _x0, _y0, _theta;
    private:
        double _A0, _A1, _B0, _B1, _AB0, _AB1, _BC0, _BC1;
        double _ABAB, _BCBC;
    };

    class ObscRay2 : public Obscuration2 {
    public:
        ObscRay2(double width, double theta, double x0=0.0, double y0=0.0);
        Obscuration2* getDevPtr() const override;
        bool contains(double x, double y) const override;

        const double _width, _theta, _x0, _y0;
    private:
        const ObscRectangle2 _rect;
    };

    class ObscNegation2 : public Obscuration2 {
    public:
        ObscNegation2(Obscuration2* original);
        Obscuration2* getDevPtr() const override;
        bool contains(double x, double y) const override;

        Obscuration2* _original;
    };

    class ObscUnion2 : public Obscuration2 {
    public:
        ObscUnion2(Obscuration2** obscs, size_t nobsc);
        Obscuration2* getDevPtr() const override;
        bool contains(double x, double y) const override;

        size_t _nobsc;
        Obscuration2** _obscs;
    };

    class ObscIntersection2 : public Obscuration2 {
    public:
        ObscIntersection2(Obscuration2** obscs, size_t nobsc);
        Obscuration2* getDevPtr() const override;
        bool contains(double x, double y) const override;

        size_t _nobsc;
        Obscuration2** _obscs;
    };
}

#endif
