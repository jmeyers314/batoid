#ifndef batoid_medium2_h
#define batoid_medium2_h

#include "dualView.h"
#include <memory>
#include <array>
#include <cmath>

namespace batoid {

    class Medium2 {
    public:
        Medium2() : _devPtr(nullptr) {}
        virtual Medium2* getDevPtr() const = 0;
        virtual ~Medium2() {}

        virtual double getN(double) const = 0;

        void getNMany(double* wavelength, size_t size, double* out) const;
        void getNMany(const DualView<double>& wavelength, DualView<double>& out) const;
    protected:
        mutable Medium2* _devPtr;
    };


    class ConstMedium2 : public Medium2 {
    public:
        ConstMedium2(double n);
        virtual Medium2* getDevPtr() const override;

        virtual double getN(double) const override;
    private:
        const double _n;
    };


    class SellmeierMedium2 : public Medium2 {
    public:
        SellmeierMedium2(double B1, double B2, double B3, double C1, double C2, double C3);
        virtual Medium2* getDevPtr() const override;

        virtual double getN(double) const override;
    private:
        const double _B1, _B2, _B3, _C1, _C2, _C3;
    };


    class SumitaMedium2 : public Medium2 {
    public:
        SumitaMedium2(double A0, double A1, double A2, double A3, double A4, double A5);
        virtual Medium2* getDevPtr() const override;

        virtual double getN(double) const override;
    private:
        const double _A0, _A1, _A2, _A3, _A4, _A5;
    };


    class Air2 : public Medium2 {
    public:
        Air2(double pressure=69.328, double temperature=293.15, double h2o_pressure=1.067);
        virtual Medium2* getDevPtr() const override;

        virtual double getN(double) const override;
    private:
        const double _pressure, _temperature, _h2o_pressure; // input stored to enable cloning.
        const double _P, _T, _W;  // same as input, but transformed to better units
    };
}

#endif
