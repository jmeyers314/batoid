#include "asphere.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportAsphere(py::module& m) {
        py::class_<Asphere, std::shared_ptr<Asphere>, Surface>(m, "Asphere")
            .def(py::init<double,double,std::vector<double>,double,double,double>(), "init",
                 "R"_a, "K"_a, "alpha"_a, "B"_a,
                 "Rin"_a=0.0, "Rout"_a=std::numeric_limits<double>::infinity())
            .def_property_readonly("R", &Asphere::getR)
            .def_property_readonly("kappa", &Asphere::getKappa)
            .def_property_readonly("alpha", &Asphere::getAlpha)
            .def_property_readonly("B", &Asphere::getB)
            .def("__call__", &Asphere::operator())
            .def("normal", &Asphere::normal)
            .def("intersect", (Intersection (Asphere::*)(const Ray&) const) &Asphere::intersect)
            .def("intersect", (std::vector<jtrace::Intersection> (Asphere::*)(const std::vector<jtrace::Ray>&) const) &Asphere::intersect)
            .def("__repr__", &Asphere::repr);
    }
}
