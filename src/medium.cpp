#include "medium.h"
#include <cmath>

namespace jtrace {
    ConstMedium::ConstMedium(double _n) : n(_n) {}

    double ConstMedium::getN(double wavelength) const {
        return n;
    }

    TableMedium::TableMedium(std::shared_ptr<Table<double,double>> _table) :
        table(_table) {}

    double TableMedium::getN(double wavelength) const {
        return (*table)(wavelength);
    }

    SellmeierMedium::SellmeierMedium(
        double _B1, double _B2, double _B3,
        double _C1, double _C2, double _C3) :
        B1(_B1), B2(_B2), B3(_B3), C1(_C1), C2(_C2), C3(_C3) {}

    double SellmeierMedium::getN(double wavelength) const {
        // Sellmeier coefficients assume wavelength is in microns, so we have to multiply (1e6)**2
        double x = wavelength*wavelength*1e12;
        return std::sqrt(1.0 + B1*x/(x-C1) + B2*x/(x-C2) + B3*x/(x-C3));
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
    Air::Air(double _p, double _t, double _h2o) :
        P(_p * 7.50061683), T(_t - 273.15), W(_h2o * 7.50061683) {}

    double Air::getN(double wavelength) const {
        // inverse wavenumber squared in micron^-2
        double sigma_squared = 1e-12 / (wavelength*wavelength);
        double n_minus_one = (64.328 + (29498.1 / (146.0 - sigma_squared))
                              + (255.4 / (41.0 - sigma_squared))) * 1.e-6;
        n_minus_one *= P * (1.0 + (1.049 - 0.0157 * T) * 1.e-6 * P) / (720.883 * (1.0 + 0.003661 * T));
        n_minus_one -= (0.0624 - 0.000680 * sigma_squared)/(1.0 + 0.003661 * T) * W * 1.e-6;
        return 1+n_minus_one;
    }

}
