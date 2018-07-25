#include "asphere.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportAsphere(py::module& m) {
        py::class_<Asphere, std::shared_ptr<Asphere>, Surface>(m, "Asphere")
            .def(py::init<double,double,std::vector<double>>(), "init", "R"_a, "conic"_a, "coefs"_a)
            .def_property_readonly("R", &Asphere::getR)
            .def_property_readonly("conic", &Asphere::getConic)
            .def_property_readonly("coefs", &Asphere::getCoefs);
    }
}
