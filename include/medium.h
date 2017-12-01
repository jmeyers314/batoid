#ifndef batoid_medium_h
#define batoid_medium_h

#include "table.h"
#include <memory>
#include <array>

namespace batoid {
    class Medium {
    public:
        virtual double getN(double wavelength) const = 0;
    };


    class ConstMedium : public Medium {
    public:
        ConstMedium(double n);
        double getN(double wavelength) const override;
    private:
        const double _n;
    };
    bool operator==(const ConstMedium& cm1, const ConstMedium& cm2);
    bool operator!=(const ConstMedium& cm1, const ConstMedium& cm2);


    class TableMedium : public Medium {
    public:
        TableMedium(std::shared_ptr<Table<double,double>> table);
        double getN(double wavelength) const override;
        std::shared_ptr<Table<double,double>> getTable() const;
    private:
        const std::shared_ptr<Table<double,double>> _table;
    };
    bool operator==(const TableMedium& tm1, const TableMedium& tm2);
    bool operator!=(const TableMedium& tm1, const TableMedium& tm2);


    class SellmeierMedium : public Medium {
    public:
        SellmeierMedium(double B1, double B2, double B3, double C1, double C2, double C3);
        SellmeierMedium(std::array<double,6>);
        double getN(double wavelength) const override;
        std::array<double,6> getCoefs() const;
    private:
        const double _B1, _B2, _B3, _C1, _C2, _C3;
    };
    bool operator==(const SellmeierMedium& sm1, const SellmeierMedium& sm2);
    bool operator!=(const SellmeierMedium& sm1, const SellmeierMedium& sm2);


    class Air: public Medium {
    public:
        Air(double pressure=69.328, double temperature=293.15, double h2o_pressure=1.067);
        double getN(double wavelength) const override;
        double getPressure() const { return _pressure; }
        double getTemperature() const { return _temperature; }
        double getH2OPressure() const { return _h2o_pressure; }
    private:
        const double _pressure, _temperature, _h2o_pressure; // input vars
        const double _P, _T, _W;  // same, but transformed to better units
    };
    bool operator==(const Air& a1, const Air& a2);
    bool operator!=(const Air& a1, const Air& a2);


}

#endif
