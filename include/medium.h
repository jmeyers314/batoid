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
}

#endif
