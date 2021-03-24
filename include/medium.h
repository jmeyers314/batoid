#ifndef batoid_medium_h
#define batoid_medium_h

#include <cstdlib>


namespace batoid {
    class Medium {
    public:
        Medium();
        virtual ~Medium();

        virtual double getN(double wavelength) const = 0;

        virtual const Medium* getDevPtr() const = 0;

    protected:
        mutable Medium* _devPtr;
    };


    class ConstMedium : public Medium {
    public:
        ConstMedium(double n);
        ~ConstMedium();

        double getN(double wavelength) const override;

        const Medium* getDevPtr() const override;

    private:
        const double _n;
    };


    class TableMedium : public Medium {
    public:
        TableMedium(const double* args, const double* vals, const size_t size);
        ~TableMedium();

        double getN(double wavelength) const override;

        const Medium* getDevPtr() const override;

    private:
        const double* _args;
        const double* _vals;
        const size_t _size;
    };


    class SellmeierMedium : public Medium {
    public:
        SellmeierMedium(double B1, double B2, double B3, double C1, double C2, double C3);
        ~SellmeierMedium();

        double getN(double wavelength) const override;

        const Medium* getDevPtr() const override;

    private:
        const double _B1, _B2, _B3, _C1, _C2, _C3;
    };


    class SumitaMedium : public Medium {
    public:
        SumitaMedium(double A0, double A1, double A2, double A3, double A4, double A5);
        ~SumitaMedium();

        double getN(double wavelength) const override;

        const Medium* getDevPtr() const override;

    private:
        const double _A0, _A1, _A2, _A3, _A4, _A5;
    };


    class Air: public Medium {
    public:
        Air(double pressure=69.328, double temperature=293.15, double h2o_pressure=1.067);
        ~Air();

        double getN(double wavelength) const override;

        const Medium* getDevPtr() const override;

    private:
        const double _pressure, _temperature, _h2o_pressure; // input vars
        const double _P, _T, _W;  // same, but transformed to better units
    };
}

#endif
