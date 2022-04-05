#ifndef batoid_obscuration_h
#define batoid_obscuration_h

#include <cstdlib>  // for size_t

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif


    /////////////////
    // Obscuration //
    /////////////////

    class Obscuration {
    public:
        Obscuration();
        virtual ~Obscuration();

        virtual bool contains(double x, double y) const = 0;
    };


    ////////////////
    // ObscCircle //
    ////////////////

    class ObscCircle : public Obscuration {
    public:
        ObscCircle(double radius, double x0=0.0, double y0=0.0);
        ~ObscCircle();

        bool contains(double x, double y) const override;

    private:
        const double _radius, _x0, _y0;
    };


    /////////////////
    // ObscAnnulus //
    /////////////////

    class ObscAnnulus : public Obscuration {
    public:
        ObscAnnulus(double inner, double outer, double x0=0.0, double y0=0.0);
        ~ObscAnnulus();

        bool contains(double x, double y) const override;

    private:
        const double _inner, _outer, _x0, _y0;
    };


    ///////////////////
    // ObscRectangle //
    ///////////////////

    class ObscRectangle : public Obscuration {
    public:
        ObscRectangle(double width, double height, double x0=0.0, double y0=0.0, double theta=0.0);
        ~ObscRectangle();

        bool contains(double x, double y) const override;

    private:
        const double _width, _height, _x0, _y0, _theta;
        const double _sth, _cth;
    };


    /////////////
    // ObscRay //
    /////////////

    class ObscRay : public Obscuration {
    public:
        ObscRay(double width, double theta, double x0=0.0, double y0=0.0);
        ~ObscRay();

        bool contains(double x, double y) const override;

    private:
        const double _width, _theta, _x0, _y0;
        const double _sth, _cth;
    };


    /////////////////
    // ObscPolygon //
    /////////////////

    class ObscPolygon : public Obscuration {
    public:
        ObscPolygon(const double* xp, const double* yp, const size_t size);
        ~ObscPolygon();

        bool contains(double x, double y) const override;

        // void containsGrid(
        //     const double* xgrid, const double* ygrid, bool* out, const size_t nx, const size_t ny
        // ) const;

    private:
        const double* _xp;
        const double* _yp;
        const size_t _size;
    };


    //////////////////
    // ObscNegation //
    //////////////////

    class ObscNegation : public Obscuration {
    public:
        ObscNegation(const Obscuration* original);
        ~ObscNegation();

        bool contains(double x, double y) const override;

    private:
        const Obscuration* _original;
    };


    ///////////////
    // ObscUnion //
    ///////////////

    class ObscUnion : public Obscuration {
    public:
        ObscUnion(const Obscuration** obscs, size_t nobsc);
        ~ObscUnion();

        bool contains(double x, double y) const override;

    private:
        const Obscuration** _obscs;
        size_t _nobsc;
    };


    //////////////////////
    // ObscIntersection //
    //////////////////////

    class ObscIntersection : public Obscuration {
    public:
        ObscIntersection(const Obscuration** obscs, size_t nobsc);
        ~ObscIntersection();

        bool contains(double x, double y) const override;

    private:
        const Obscuration** _obscs;
        size_t _nobsc;
    };


#if defined(BATOID_GPU)
    #pragma omp end declare target
#endif


    ///////////////////////
    // ObscurationHandle //
    ///////////////////////

    class ObscurationHandle {
    public:
        ObscurationHandle();

        virtual ~ObscurationHandle();

        const Obscuration* getPtr() const;

        const Obscuration* getHostPtr() const;

    protected:
        Obscuration* _hostPtr;
        Obscuration* _devicePtr;
    };


    //////////////////////
    // ObscCircleHandle //
    //////////////////////

    class ObscCircleHandle : public ObscurationHandle {
    public:
        ObscCircleHandle(double radius, double x0=0.0, double y0=0.0);
        virtual ~ObscCircleHandle();
    };


    ///////////////////////
    // ObscAnnulusHandle //
    ///////////////////////

    class ObscAnnulusHandle : public ObscurationHandle {
    public:
        ObscAnnulusHandle(double inner, double outer, double x0=0.0, double y0=0.0);
        virtual ~ObscAnnulusHandle();
    };


    /////////////////////////
    // ObscRectangleHandle //
    /////////////////////////

    class ObscRectangleHandle : public ObscurationHandle {
    public:
        ObscRectangleHandle(double width, double height, double x0=0.0, double y0=0.0, double theta=0.0);
        virtual ~ObscRectangleHandle();
    };


    ///////////////////
    // ObscRayHandle //
    ///////////////////

    class ObscRayHandle : public ObscurationHandle {
    public:
        ObscRayHandle(double width, double theta, double x0=0.0, double y0=0.0);
        ~ObscRayHandle();
    };


    ///////////////////////
    // ObscPolygonHandle //
    ///////////////////////

    class ObscPolygonHandle : public ObscurationHandle {
    public:
        ObscPolygonHandle(const double* xp, const double* yp, const size_t size);
        ~ObscPolygonHandle();
    private:
        const double* _xp;
        const double* _yp;
        const size_t _size;
    };


    ////////////////////////
    // ObscNegationHandle //
    ////////////////////////

    class ObscNegationHandle : public ObscurationHandle {
    public:
        ObscNegationHandle(const ObscurationHandle* original);
        ~ObscNegationHandle();
    private:
        const Obscuration* _original;
    };


    /////////////////////
    // ObscUnionHandle //
    /////////////////////

    class ObscUnionHandle : public ObscurationHandle {
    public:
        ObscUnionHandle(const ObscurationHandle** handles, size_t nobsc);
        ~ObscUnionHandle();

        static const Obscuration** _getObscs(
            const ObscurationHandle** handles, const size_t nobsc, bool host
        );

    private:
        const Obscuration** _hostObscs;
        const Obscuration** _devObscs;
        size_t _nobsc;
    };


    ////////////////////////////
    // ObscIntersectionHandle //
    ////////////////////////////

    class ObscIntersectionHandle : public ObscurationHandle {
    public:
        ObscIntersectionHandle(const ObscurationHandle** handles, size_t nobsc);
        ~ObscIntersectionHandle();

        static const Obscuration** _getObscs(
            const ObscurationHandle** handles, const size_t nobsc, bool host
        );

    private:
        const Obscuration** _hostObscs;
        const Obscuration** _devObscs;
        size_t _nobsc;
    };

}

#endif
