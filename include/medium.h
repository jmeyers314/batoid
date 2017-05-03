#ifndef jtrace_medium_h
#define jtrace_medium_h

#include "table.h"
#include <memory>

namespace jtrace {
    class Medium {
    public:
        virtual double getN(double wavelength) const = 0;
    };

    class ConstMedium : public Medium {
    public:
        ConstMedium(double n);
        double getN(double wavelength) const override;
    private:
        const double n;
    };

    class TableMedium : public Medium {
    public:
        TableMedium(std::shared_ptr<Table<double,double>> table);
        double getN(double wavelength) const override;
    private:
        const std::shared_ptr<Table<double,double>> table;
    };

    class SellmeierMedium : public Medium {
    public:
        SellmeierMedium(double B1, double B2, double B3, double C1, double C2, double C3);
        double getN(double wavelength) const override;
    private:
        const double B1, B2, B3, C1, C2, C3;
    };

    class Air: public Medium {
    public:
        Air(double pressure=69.328, double temperature=293.15, double H2O_pressure=1.067);
        double getN(double wavelength) const override;
    private:
        const double P, T, W;
    };

}

#endif
