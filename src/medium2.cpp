#include "medium2.h"
#include <cmath>
#include <sstream>

namespace batoid {

    template<typename T>
    void Medium2CRTP<T>::getNMany(size_t size, double* wavelength, double* out) const {
        DualView<double> wdv(wavelength, size);
        DualView<double> outdv(out, size, DVOwnerType::device);
        getNMany(wdv, outdv);
        outdv.syncToHost();
    }

    template class Medium2CRTP<SellmeierMedium2>;
}
