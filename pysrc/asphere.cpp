#include "asphere.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportAsphere(py::module& m) {
        py::class_<Asphere, std::shared_ptr<Asphere>, Surface>(m, "Asphere")
            .def(py::init<double,double,std::vector<double>,double,double,double>(), "init",
                 "R"_a, "K"_a, "alpha"_a, "B"_a,
                 "Rin"_a=0.0, "Rout"_a=std::numeric_limits<double>::infinity())
            .def_property_readonly("R", &Asphere::getR)
            .def_property_readonly("kappa", &Asphere::getKappa)
            .def_property_readonly("alpha", &Asphere::getAlpha)
            .def_property_readonly("B", &Asphere::getB)
            .def_property_readonly("Rin", &Asphere::getRin)
            .def_property_readonly("Rout", &Asphere::getRout);
    }
}
