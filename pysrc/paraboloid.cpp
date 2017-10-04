#include "paraboloid.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportParaboloid(py::module& m) {
        py::class_<Paraboloid, std::shared_ptr<Paraboloid>, Surface>(m, "Paraboloid")
            .def(py::init<double,double,double,double>(), "init",
                 "R"_a, "B"_a,
                 "Rin"_a=0.0, "Rout"_a=std::numeric_limits<double>::infinity())
            .def_property_readonly("R", &Paraboloid::getR)
            .def_property_readonly("B", &Paraboloid::getB)
            .def_property_readonly("Rin", &Paraboloid::getRin)
            .def_property_readonly("Rout", &Paraboloid::getRout)
            .def("sag", py::vectorize(&Paraboloid::sag))
            .def("normal", &Paraboloid::normal)
            .def("intersect", (Intersection (Paraboloid::*)(const Ray&) const) &Paraboloid::intersect)
            .def("intersect", (std::vector<batoid::Intersection> (Paraboloid::*)(const std::vector<batoid::Ray>&) const) &Paraboloid::intersect)
            .def("__repr__", &Paraboloid::repr);
    }
}
