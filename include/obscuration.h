#ifndef batoid_obscuration_h
#define batoid_obscuration_h

#include <cstdlib>  // for size_t

namespace batoid {
    class Obscuration {
    public:
        Obscuration();
        virtual ~Obscuration();

        virtual bool contains(double x, double y) const = 0;

        virtual const Obscuration* getDevPtr() const = 0;

    protected:
        mutable Obscuration* _devPtr;
    };


    class ObscCircle : public Obscuration {
    public:
        ObscCircle(double radius, double x0=0.0, double y0=0.0);
        ~ObscCircle();

        bool contains(double x, double y) const override;

        const Obscuration* getDevPtr() const override;

    private:
        const double _radius, _x0, _y0;
    };


    class ObscAnnulus : public Obscuration {
    public:
        ObscAnnulus(double inner, double outer, double x0=0.0, double y0=0.0);
        ~ObscAnnulus();

        bool contains(double x, double y) const override;

        const Obscuration* getDevPtr() const override;

    private:
        const double _inner, _outer, _x0, _y0;
    };


    class ObscRectangle : public Obscuration {
    public:
        ObscRectangle(double width, double height, double x0=0.0, double y0=0.0, double theta=0.0);
        ~ObscRectangle();

        bool contains(double x, double y) const override;

        const Obscuration* getDevPtr() const override;

    private:
        const double _width, _height, _x0, _y0, _theta;
        const double _sth, _cth;
    };


    class ObscRay : public Obscuration {
    public:
        ObscRay(double width, double theta, double x0=0.0, double y0=0.0);
        ~ObscRay();

        bool contains(double x, double y) const override;

        const Obscuration* getDevPtr() const override;

    private:
        const double _width, _theta, _x0, _y0;
        const double _sth, _cth;
    };


    class ObscPolygon : public Obscuration {
    public:
        ObscPolygon(const double* xp, const double* yp, const size_t size);
        ~ObscPolygon();

        bool contains(double x, double y) const override;

        void containsGrid(
            const double* xgrid, const double* ygrid, bool* out, const size_t nx, const size_t ny
        ) const;

        const Obscuration* getDevPtr() const override;

    private:
        const double* _xp;
        const double* _yp;
        const size_t _size;

        static double* _copyArr(const double* coefs, const size_t size);
    };


    class ObscNegation : public Obscuration {
    public:
        ObscNegation(const Obscuration* original);
        ~ObscNegation();

        bool contains(double x, double y) const override;

        const Obscuration* getDevPtr() const override;

    private:
        const Obscuration* _original;
    };


    class ObscUnion : public Obscuration {
    public:
        ObscUnion(const Obscuration** obscs, size_t nobsc);
        ~ObscUnion();

        bool contains(double x, double y) const override;

        const Obscuration* getDevPtr() const override;

    private:
        const Obscuration** _obscs;
        size_t _nobsc;
    };


    class ObscIntersection : public Obscuration {
    public:
        ObscIntersection(const Obscuration** obscs, size_t nobsc);
        ~ObscIntersection();

        bool contains(double x, double y) const override;

        const Obscuration* getDevPtr() const override;

    private:
        const Obscuration** _obscs;
        size_t _nobsc;
    };
}

#endif
