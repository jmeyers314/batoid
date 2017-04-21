#include "medium.h"

namespace jtrace {
    ConstMedium::ConstMedium(double _n) : n(_n) {}

    double ConstMedium::getN(double wavelength) const {
        return n;
    }

    TableMedium::TableMedium(std::shared_ptr<Table<double,double>> _table) : table(_table) {}

    double TableMedium::getN(double wavelength) const {
        return (*table)(wavelength);
    }
}
