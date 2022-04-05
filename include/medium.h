#ifndef batoid_medium_h
#define batoid_medium_h

#include <cstdlib>


namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif


    ////////////
    // Medium //
    ////////////

    class Medium {
    public:
        Medium();
        virtual ~Medium();

        virtual double getN(const double wavelength) const = 0;
    };


    /////////////////
    // ConstMedium //
    /////////////////

    class ConstMedium : public Medium {
    public:
        ConstMedium(const double n);
        ~ConstMedium();

        double getN(const double wavelength) const override;

    private:
        const double _n;
    };


    /////////////////
    // TableMedium //
    /////////////////

    class TableMedium : public Medium {
    public:
        TableMedium(const double* args, const double* vals, const size_t size);
        ~TableMedium();

        double getN(const double wavelength) const override;

    private:
        const double* _args;
        const double* _vals;
        const size_t _size;
    };


    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    /////////////////////
    // SellmeierMedium //
    /////////////////////

    class SellmeierMedium : public Medium {
    public:
        SellmeierMedium(double B1, double B2, double B3, double C1, double C2, double C3);
        ~SellmeierMedium();

        double getN(const double wavelength) const override;

    private:
        const double _B1, _B2, _B3, _C1, _C2, _C3;
    };


    //////////////////
    // SumitaMedium //
    //////////////////

    class SumitaMedium : public Medium {
    public:
        SumitaMedium(double A0, double A1, double A2, double A3, double A4, double A5);
        ~SumitaMedium();

        double getN(const double wavelength) const override;

    private:
        const double _A0, _A1, _A2, _A3, _A4, _A5;
    };


    class Air : public Medium {
    public:
        Air(double pressure=69.328, double temperature=293.15, double h2o_pressure=1.067);
        ~Air();

        double getN(const double wavelength) const override;

    private:
        const double _pressure, _temperature, _h2o_pressure; // input vars
        const double _P, _T, _W;  // same, but transformed to better units
    };


    //////////////////
    // MediumHandle //
    //////////////////

    class MediumHandle {
    public:
        MediumHandle();

        virtual ~MediumHandle();

        const Medium* getPtr() const;

        const Medium* getHostPtr() const;

    protected:
        Medium* _hostPtr;
        Medium* _devicePtr;
    };


    ///////////////////////
    // ConstMediumHandle //
    ///////////////////////

    class ConstMediumHandle : public MediumHandle {
    public:
        ConstMediumHandle(const double n);
        virtual ~ConstMediumHandle();
    };


    ///////////////////////
    // TableMediumHandle //
    ///////////////////////

    class TableMediumHandle : public MediumHandle {
    public:
        TableMediumHandle(const double* args, const double* vals, const size_t size);
        ~TableMediumHandle();
    private:
        const double* _args;
        const double* _vals;
        const size_t _size;

    };


    ///////////////////////////
    // SellmeierMediumHandle //
    ///////////////////////////

    class SellmeierMediumHandle : public MediumHandle {
    public:
        SellmeierMediumHandle(double B1, double B2, double B3, double C1, double C2, double C3);
        ~SellmeierMediumHandle();
    };


    ////////////////////////
    // SumitaMediumHandle //
    ////////////////////////

    class SumitaMediumHandle : public MediumHandle {
    public:
        SumitaMediumHandle(double A0, double A1, double A2, double A3, double A4, double A5);
        ~SumitaMediumHandle();
    };


    ///////////////
    // AirHandle //
    ///////////////

    class AirHandle: public MediumHandle {
    public:
        AirHandle(double pressure=69.328, double temperature=293.15, double h2o_pressure=1.067);
        ~AirHandle();
    };

}

#endif
