#include "zernike.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace batoid {
    void pyExportZernike(py::module& m) {
        m.def("nCr", &zernike::nCr);
        m.def("binomial", &zernike::binomial);
        m.def("noll_to_zern", &zernike::noll_to_zern);
        m.def("_zern_rho_coefs", &zernike::_zern_rho_coefs);
        m.def("h", &zernike::h);
        m.def("Q", &zernike::Q);
        m.def("Q0", &zernike::Q0);
        m.def("_annular_zern_rho_coefs", &zernike::_annular_zern_rho_coefs);
    }
}
