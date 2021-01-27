#include "medium.h"
#include <new>
#include <cmath>
#include <cstdio>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        Medium::Medium() :
            _devPtr(nullptr)
        {}

        Medium::~Medium() {}


        ConstMedium::ConstMedium(double n) :
            Medium(), _n(n)
        {}

        ConstMedium::~ConstMedium() {}

        double ConstMedium::getN(double wavelength) const {
            return _n;
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    const Medium* ConstMedium::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            Medium* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new ConstMedium(_n);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        TableMedium::TableMedium(
            const double* args, const double* vals, const size_t size
        ) :
            Medium(), _args(args), _vals(vals), _size(size)
        {}

        TableMedium::~TableMedium() {
            #if defined(BATOID_GPU)
                if (_devPtr) {
                    Medium* ptr = _devPtr;
                    #pragma omp target is_device_ptr(ptr)
                    {
                        delete ptr;
                    }
                    const double* args = _args;
                    const double* vals = _vals;
                    #pragma omp target exit data \
                        map(release:args[:_size], vals[:_size])
                }
            #endif
        }

        double TableMedium::getN(double wavelength) const {
            // Linear search.  Better for GPU's?  and not that painful for small arrays?
            if (wavelength < _args[0])
                return NAN;
            if (wavelength > _args[_size-1])
                return NAN;
            int upperIdx;
            for(upperIdx=1; upperIdx<_size; upperIdx++) {
                if (wavelength < _args[upperIdx])
                    break;
            }
            double out = (wavelength - _args[upperIdx-1]);
            out *= (_vals[upperIdx] - _vals[upperIdx-1]);
            out /= (_args[upperIdx] - _args[upperIdx-1]);
            out += _vals[upperIdx-1];
            return out;
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Medium* TableMedium::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (!_devPtr) {
                Medium* ptr;
                // Allocate arrays on device
                const double* args = _args;
                const double* vals = _vals;
                #pragma omp target enter data \
                    map(to:args[:_size], vals[:_size])
                #pragma omp target map(from:ptr)
                {
                    ptr = new TableMedium(args, vals, _size);
                }
                _devPtr = ptr;
            }
            return _devPtr;
        #else
            return this;
        #endif
    }

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        SellmeierMedium::SellmeierMedium(
            double B1, double B2, double B3,
            double C1, double C2, double C3
        ) :
            Medium(), _B1(B1), _B2(B2), _B3(B3), _C1(C1), _C2(C2), _C3(C3)
        {}

        SellmeierMedium::~SellmeierMedium() {}

        double SellmeierMedium::getN(double wavelength) const {
            // Sellmeier coefficients assume wavelength is in microns, so we have to multiply (1e6)**2
            double x = wavelength*wavelength*1e12;
            return std::sqrt(1.0 + _B1*x/(x-_C1) + _B2*x/(x-_C2) + _B3*x/(x-_C3));
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif


    const Medium* SellmeierMedium::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            Medium* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new SellmeierMedium(_B1, _B2, _B3, _C1, _C2, _C3);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        SumitaMedium::SumitaMedium(
            double A0, double A1, double A2,
            double A3, double A4, double A5
        ) :
            Medium(), _A0(A0), _A1(A1), _A2(A2), _A3(A3), _A4(A4), _A5(A5)
        {}

        SumitaMedium::~SumitaMedium() {}

        double SumitaMedium::getN(double wavelength) const {
            //Sumita coefficients assume wavelength is in microns, so we have to multiply (1e6)**2
            double x = wavelength*wavelength*1e12;
            double y = 1./x;
            return std::sqrt(_A0 + _A1*x + y*(_A2 + y*(_A3 + y*(_A4 + y*_A5))));
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Medium* SumitaMedium::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            Medium* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new SumitaMedium(_A0, _A1, _A2, _A3, _A4, _A5);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }


    // Uses the formulae given in Filippenko (1982), which appear to come from Edlen (1953),
    // and Coleman, Bozman, and Meggers (1960).  The units of the original formula are non-SI,
    // being mmHg for pressure (and water vapor pressure), and degrees C for temperature.  This
    // class accepts SI units, however, and transforms them when plugging into the formula.
    //
    // The default values for temperature, pressure and water vapor pressure are expected to be
    // appropriate for LSST at Cerro Pachon, Chile, but they are broadly reasonable for most
    // observatories.
    //
    // @param pressure       Air pressure in kiloPascals.
    // @param temperature    Temperature in Kelvins.
    // @param H2O_pressure   Water vapor pressure in kiloPascals.
    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif

        Air::Air(double pressure, double temperature, double h2o_pressure) :
            _pressure(pressure), _temperature(temperature), _h2o_pressure(h2o_pressure),
            _P(pressure * 7.50061683), _T(temperature - 273.15), _W(h2o_pressure * 7.50061683)
        {}

        Air::~Air() {}

        double Air::getN(double wavelength) const {
            // inverse wavenumber squared in micron^-2
            double sigma_squared = 1e-12 / (wavelength*wavelength);
            double n_minus_one = (64.328 + (29498.1 / (146.0 - sigma_squared))
                                  + (255.4 / (41.0 - sigma_squared))) * 1.e-6;
            n_minus_one *= _P * (1.0 + (1.049 - 0.0157 * _T) * 1.e-6 * _P) / (720.883 * (1.0 + 0.003661 * _T));
            n_minus_one -= (0.0624 - 0.000680 * sigma_squared)/(1.0 + 0.003661 * _T) * _W * 1.e-6;
            return 1+n_minus_one;
        }

    #if defined(BATOID_GPU)
        #pragma omp end declare target
    #endif

    const Medium* Air::getDevPtr() const {
        #if defined(BATOID_GPU)
            if (_devPtr)
                return _devPtr;
            Medium* ptr;
            #pragma omp target map(from:ptr)
            {
                ptr = new Air(_pressure, _temperature, _h2o_pressure);
            }
            _devPtr = ptr;
            return ptr;
        #else
            return this;
        #endif
    }

}
