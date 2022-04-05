#include "medium.h"
#include <new>
#include <cmath>
#include <cstdio>
#include <omp.h>

namespace batoid {

    #if defined(BATOID_GPU)
        #pragma omp declare target
    #endif


    ////////////
    // Medium //
    ////////////

    Medium::Medium() {}

    Medium::~Medium() {}


    /////////////////
    // ConstMedium //
    /////////////////

    ConstMedium::ConstMedium(const double n) : _n(n) {}

    ConstMedium::~ConstMedium() {}

    double ConstMedium::getN(const double wavelength) const {
        return _n;
    }


    /////////////////
    // TableMedium //
    /////////////////

    TableMedium::TableMedium(
        const double* args, const double* vals, const size_t size
    ) :
        Medium(), _args(args), _vals(vals), _size(size)
    {}

    TableMedium::~TableMedium() {}

    double TableMedium::getN(const double wavelength) const {
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


    /////////////////////
    // SellmeierMedium //
    /////////////////////

    SellmeierMedium::SellmeierMedium(
        double B1, double B2, double B3,
        double C1, double C2, double C3
    ) :
        Medium(), _B1(B1), _B2(B2), _B3(B3), _C1(C1), _C2(C2), _C3(C3)
    {}

    SellmeierMedium::~SellmeierMedium() {}

    double SellmeierMedium::getN(const double wavelength) const {
        // Sellmeier coefficients assume wavelength is in microns, so we have to multiply (1e6)**2
        double x = wavelength*wavelength*1e12;
        return std::sqrt(1.0 + _B1*x/(x-_C1) + _B2*x/(x-_C2) + _B3*x/(x-_C3));
    }


    /////////////////////
    // SumitaMedium //
    /////////////////////

    SumitaMedium::SumitaMedium(
        double A0, double A1, double A2,
        double A3, double A4, double A5
    ) :
        Medium(), _A0(A0), _A1(A1), _A2(A2), _A3(A3), _A4(A4), _A5(A5)
    {}

    SumitaMedium::~SumitaMedium() {}

    double SumitaMedium::getN(const double wavelength) const {
        //Sumita coefficients assume wavelength is in microns, so we have to multiply (1e6)**2
        double x = wavelength*wavelength*1e12;
        double y = 1./x;
        return std::sqrt(_A0 + _A1*x + y*(_A2 + y*(_A3 + y*(_A4 + y*_A5))));
    }


    /////////
    // Air //
    /////////

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

    Air::Air(double pressure, double temperature, double h2o_pressure) :
        _pressure(pressure), _temperature(temperature), _h2o_pressure(h2o_pressure),
        _P(pressure * 7.50061683), _T(temperature - 273.15), _W(h2o_pressure * 7.50061683)
    {}

    Air::~Air() {}

    double Air::getN(const double wavelength) const {
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


    //////////////////
    // MediumHandle //
    //////////////////

    MediumHandle::MediumHandle() :
        _hostPtr(nullptr),
        _devicePtr(nullptr)
    {}

    MediumHandle::~MediumHandle() {}

    const Medium* MediumHandle::getPtr() const {
        #if defined(BATOID_GPU)
            return _devicePtr;
        #else
            return _hostPtr;
        #endif
    }

    const Medium* MediumHandle::getHostPtr() const {
        return _hostPtr;
    }


    ///////////////////////
    // ConstMediumHandle //
    ///////////////////////

    ConstMediumHandle::ConstMediumHandle(const double n) :
        MediumHandle()
    {
        _hostPtr = new ConstMedium(n);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(ConstMedium), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) ConstMedium(n);
            }
        #endif
    }

    ConstMediumHandle::~ConstMediumHandle() {
        #if defined(BATOID_GPU)
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    ///////////////////////
    // TableMediumHandle //
    ///////////////////////

    TableMediumHandle::TableMediumHandle(const double* args, const double* vals, const size_t size) :
        MediumHandle(),
        _args(args), _vals(vals), _size(size)
    {
        _hostPtr = new TableMedium(args, vals, size);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(TableMedium), omp_get_default_device());
            const size_t lcl_size = _size;
            const double* lcl_args = _args;
            const double* lcl_vals = _vals;
            #pragma omp target enter data map(to:lcl_args[:lcl_size], lcl_vals[:lcl_size])
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) TableMedium(lcl_args, lcl_vals, lcl_size);
            }
        #endif
    }

    TableMediumHandle::~TableMediumHandle() {
        #if defined(BATOID_GPU)
            #pragma omp target exit data \
                map(release:_args[:_size], _vals[:_size])
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    ///////////////////////////
    // SellmeierMediumHandle //
    ///////////////////////////

    SellmeierMediumHandle::SellmeierMediumHandle(
        double B1, double B2, double B3,
        double C1, double C2, double C3
    ) :
        MediumHandle()
    {
        _hostPtr = new SellmeierMedium(B1, B2, B3, C1, C2, C3);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(SellmeierMedium), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) SellmeierMedium(B1, B2, B3, C1, C2, C3);
            }
        #endif
    }

    SellmeierMediumHandle::~SellmeierMediumHandle() {
        #if defined(BATOID_GPU)
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }


    ////////////////////////
    // SumitaMediumHandle //
    ////////////////////////

    SumitaMediumHandle::SumitaMediumHandle(
        double A0, double A1, double A2,
        double A3, double A4, double A5
    ) :
        MediumHandle()
    {
        _hostPtr = new SumitaMedium(A0, A1, A2, A3, A4, A5);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(SumitaMedium), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) SumitaMedium(A0, A1, A2, A3, A4, A5);
            }
        #endif
    }

    SumitaMediumHandle::~SumitaMediumHandle() {
        #if defined(BATOID_GPU)
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }

    AirHandle::AirHandle(double pressure, double temperature, double h2o_pressure) :
        MediumHandle()
    {
        _hostPtr = new Air(pressure, temperature, h2o_pressure);
        #if defined(BATOID_GPU)
            auto alloc = omp_target_alloc(sizeof(Air), omp_get_default_device());
            #pragma omp target map(from:_devicePtr), is_device_ptr(alloc)
            {
                _devicePtr = new (alloc) Air(pressure, temperature, h2o_pressure);
            }
        #endif
    }

    AirHandle::~AirHandle() {
        #if defined(BATOID_GPU)
            omp_target_free(_devicePtr, omp_get_default_device());
        #endif
        delete _hostPtr;
    }
}
