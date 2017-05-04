#include "quadric.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportQuadric(py::module& m) {
        py::class_<Quadric, std::shared_ptr<Quadric>, Surface>(m, "Quadric")
            .def(py::init<double,double,double>(), "init", "R"_a, "K"_a, "B"_a)
            .def_property_readonly("R", &Quadric::getR)
            .def_property_readonly("kappa", &Quadric::getKappa)
            .def_property_readonly("B", &Quadric::getB)
            .def("__call__", &Quadric::operator())
            .def("normal", &Quadric::normal)
            .def("intersect", (Intersection (Quadric::*)(const Ray&) const) &Quadric::intersect)
            .def("intersect", (std::vector<jtrace::Intersection> (Quadric::*)(const std::vector<jtrace::Ray>&) const) &Quadric::intersect)
            .def("__repr__", &Quadric::repr);
    }
}
