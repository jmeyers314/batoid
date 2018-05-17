#include "zernike.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportZernike(py::module& m) {
        m.def("nCr", &zernike::nCr);
        m.def("binomial", &zernike::binomial);
        m.def("horner2d", &zernike::horner2d);
        m.def("noll_to_zern", &zernike::noll_to_zern);
        m.def("zern_rho_coefs", &zernike::zern_rho_coefs);
        m.def("h", &zernike::h);
        m.def("Q", &zernike::Q);
        m.def("Q0", &zernike::Q0);
        m.def("annular_zern_rho_coefs", &zernike::annular_zern_rho_coefs);
        m.def("zern_coef_array", &zernike::zern_coef_array);
        m.def("xy_contribution", &zernike::xy_contribution);
        m.def("noll_coef_array", &zernike::noll_coef_array);
        m.def("rrsq_to_xy", &zernike::rrsq_to_xy);
        m.def("noll_coef_array_xy", &zernike::noll_coef_array_xy);

        py::class_<Zernike, std::shared_ptr<Zernike>, Surface>(m, "Zernike")
            .def(py::init<std::vector<double>,double,double>(), "init", "coefs"_a, "R_outer"_a=1.0, "R_inner"_a=0.0)
            .def_property_readonly("coefs", &Zernike::getCoefs)
            .def_property_readonly("R_outer", &Zernike::getROuter)
            .def_property_readonly("R_inner", &Zernike::getRInner);
    }
}
