#ifndef batoid_medium2_h
#define batoid_medium2_h

#include "dualView.h"
#include <memory>
#include <array>
#include <cmath>

namespace batoid {
    class Medium2 {
    public:
        virtual ~Medium2() {}
        virtual void getNMany(size_t size, double* wavelength, double* out) const = 0;
        virtual void getNMany(const DualView<double>& wavelength, DualView<double>& out) const = 0;
    };

    template<typename T>
    class Medium2CRTP : public Medium2 {
    public:
        virtual void getNMany(size_t size, double* wavelength, double* out) const override;
        virtual void getNMany(const DualView<double>& wavelength, DualView<double>& out) const override {
            size_t size = wavelength.size;
            double* outptr = out.deviceData;
            wavelength.syncToDevice();
            double* wptr = wavelength.deviceData;
            const T* self = static_cast<const T*>(this);
            #pragma omp target is_device_ptr(outptr, wptr) map(to:self[:1])
            {
                #pragma omp teams distribute parallel for
                for(int i=0; i<size; i++)
                    outptr[i] = self->getN(wptr[i]);
            }
        }
    };

    // class ConstMedium2 : public Medium2CRTP<ConstMedium2> {
    // public:
    //     ConstMedium2(double n) : _n(n) {}
    //     double getN(double wavelength) const { return _n; }
    // private:
    //     const double _n;
    // };

    class SellmeierMedium2 : public Medium2CRTP<SellmeierMedium2> {
    public:
        SellmeierMedium2(double B1, double B2, double B3, double C1, double C2, double C3) :
            _B1(B1), _B2(B2), _B3(B3), _C1(C1), _C2(C2), _C3(C3) {}
        double getN(double wavelength) const {
            // Sellmeier coefficients assume wavelength is in microns, so we have to multiply (1e6)**2
            double x = wavelength*wavelength*1e12;
            return sqrt(1.0 + _B1*x/(x-_C1) + _B2*x/(x-_C2) + _B3*x/(x-_C3));
        }
    private:
        const double _B1, _B2, _B3, _C1, _C2, _C3;
    };

    // class SumitaMedium2 : public Medium2CRTP<SumitaMedium2> {
    // public:
    //     SumitaMedium2(double A0, double A1, double A2, double A3, double A4, double A5) :
    //         _A0(A0), _A1(A1), _A2(A2), _A3(A3), _A4(A4), _A5(A5) {}
    //     double getN(double wavelength) const {
    //         //Sumita coefficients assume wavelength is in microns, so we have to multiply (1e6)**2
    //         double x = wavelength*wavelength*1e12;
    //         double y = 1./x;
    //         return sqrt(_A0 + _A1*x + y*(_A2 + y*(_A3 + y*(_A4 + y*_A5))));
    //     }
    // private:
    //     const double _A0, _A1, _A2, _A3, _A4, _A5;
    // };
    //
    // class Air2 : public Medium2CRTP<Air2> {
    // public:
    //     Air2(double pressure=69.328, double temperature=293.15, double h2o_pressure=1.067) :
    //         _P(pressure * 7.50061683), _T(temperature - 273.15), _W(h2o_pressure * 7.50061683) {}
    //     double getN(double wavelength) const {
    //         // inverse wavenumber squared in micron^-2
    //         double sigma_squared = 1e-12 / (wavelength*wavelength);
    //         double n_minus_one = (64.328 + (29498.1 / (146.0 - sigma_squared))
    //                               + (255.4 / (41.0 - sigma_squared))) * 1.e-6;
    //         n_minus_one *= _P * (1.0 + (1.049 - 0.0157 * _T) * 1.e-6 * _P) / (720.883 * (1.0 + 0.003661 * _T));
    //         n_minus_one -= (0.0624 - 0.000680 * sigma_squared)/(1.0 + 0.003661 * _T) * _W * 1.e-6;
    //         return 1+n_minus_one;
    //     }
    // private:
    //     const double _P, _T, _W;  // input, but transformed to better units
    // };
}

#endif
