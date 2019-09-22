#include "polynomialSurface.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;
using Eigen::MatrixXd;

namespace batoid {
    void pyExportPolynomialSurface(py::module& m) {
        py::class_<PolynomialSurface, std::shared_ptr<PolynomialSurface>, Surface>(m, "CPPPolynomialSurface")
            .def(py::init<MatrixXd>(), "init", "coefs"_a)
            .def_property_readonly("coefs", &PolynomialSurface::getCoefs);

        m.def("horner2d", &poly::horner2d);
    }
}
