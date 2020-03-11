#include "medium2.h"
#include <cmath>
#include <sstream>


#pragma omp declare target
namespace batoid {
    void Medium2::getNMany(double* wavelength, size_t size, double* out) const {
        DualView<double> wdv(wavelength, size);
        DualView<double> outdv(out, size, DVOwnerType::device);
        getNMany(wdv, outdv);
        outdv.syncToHost();
    }

    void Medium2::getNMany(const DualView<double>& wavelength, DualView<double>& out) const {
        size_t size = wavelength.size;
        double* outptr = out.deviceData;
        wavelength.syncToDevice();
        double* wptr = wavelength.deviceData;
        Medium2* devPtr = getDevPtr();
        #pragma omp target is_device_ptr(outptr, wptr, devPtr)
        {
            #pragma omp teams distribute parallel for
            for(int i=0; i<size; i++)
                outptr[i] = devPtr->getN(wptr[i]);
        }
    }



    ConstMedium2::ConstMedium2(double n) : Medium2(), _n(n) {}

    Medium2* ConstMedium2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        ConstMedium2* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new ConstMedium2(_n);
        }
        _devPtr = ptr;
        return ptr;
    }

    double ConstMedium2::getN(double wavelength) const {
        return _n;
    }



    SellmeierMedium2::SellmeierMedium2(double B1, double B2, double B3, double C1, double C2, double C3) :
        _B1(B1), _B2(B2), _B3(B3), _C1(C1), _C2(C2), _C3(C3) {}

    Medium2* SellmeierMedium2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        SellmeierMedium2* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new SellmeierMedium2(_B1, _B2, _B3, _C1, _C2, _C3);
        }
        _devPtr = ptr;
        return ptr;
    }

    double SellmeierMedium2::getN(double wavelength) const {
        double x = wavelength*wavelength*1e12;
        return sqrt(1.0 + _B1*x/(x-_C1) + _B2*x/(x-_C2) + _B3*x/(x-_C3));
    }



    SumitaMedium2::SumitaMedium2(double A0, double A1, double A2, double A3, double A4, double A5) :
        _A0(A0), _A1(A1), _A2(A2), _A3(A3), _A4(A4), _A5(A5) {}

    Medium2* SumitaMedium2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        SumitaMedium2* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new SumitaMedium2(_A0, _A1, _A2, _A3, _A4, _A5);
        }
        _devPtr = ptr;
        return ptr;
    }

    double SumitaMedium2::getN(double wavelength) const {
        double x = wavelength*wavelength*1e12;
        double y = 1./x;
        return sqrt(_A0 + _A1*x + y*(_A2 + y*(_A3 + y*(_A4 + y*_A5))));
    }



    Air2::Air2(double pressure, double temperature, double h2o_pressure) :
        _pressure(pressure), _temperature(temperature), _h2o_pressure(h2o_pressure),
        _P(pressure * 7.50061683), _T(temperature - 273.15), _W(h2o_pressure * 7.50061683) {}

    Medium2* Air2::getDevPtr() const {
        if (_devPtr)
            return _devPtr;
        Air2* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new Air2(_pressure, _temperature, _h2o_pressure);
        }
        _devPtr = ptr;
        return ptr;
    }

    double Air2::getN(double wavelength) const {
        // inverse wavenumber squared in micron^-2
        double sigma_squared = 1e-12 / (wavelength*wavelength);
        double n_minus_one = (64.328 + (29498.1 / (146.0 - sigma_squared))
                              + (255.4 / (41.0 - sigma_squared))) * 1.e-6;
        n_minus_one *= _P * (1.0 + (1.049 - 0.0157 * _T) * 1.e-6 * _P) / (720.883 * (1.0 + 0.003661 * _T));
        n_minus_one -= (0.0624 - 0.000680 * sigma_squared)/(1.0 + 0.003661 * _T) * _W * 1.e-6;
        return 1+n_minus_one;
    }
}
#pragma omp end declare target
