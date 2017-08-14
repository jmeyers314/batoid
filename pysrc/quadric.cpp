#include "quadric.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportQuadric(py::module& m) {
        py::class_<Quadric, std::shared_ptr<Quadric>, Surface>(m, "Quadric")
            .def(py::init<double,double,double,double,double>(), "init",
                 "R"_a, "K"_a, "B"_a,
                 "Rin"_a=0.0, "Rout"_a=std::numeric_limits<double>::infinity())
            .def_property_readonly("R", &Quadric::getR)
            .def_property_readonly("kappa", &Quadric::getKappa)
            .def_property_readonly("B", &Quadric::getB)
            .def("sag", &Quadric::sag)
            .def("normal", &Quadric::normal)
            .def("intersect", (Intersection (Quadric::*)(const Ray&) const) &Quadric::intersect)
            .def("intersect", (std::vector<batoid::Intersection> (Quadric::*)(const std::vector<batoid::Ray>&) const) &Quadric::intersect)
            .def("__repr__", &Quadric::repr);
    }
}
