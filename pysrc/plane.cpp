#include "plane.h"
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
    void pyExportPlane(py::module& m) {
        py::class_<Plane, std::shared_ptr<Plane>, Surface>(m, "Plane")
            .def(py::init<double,double,double>(), "init",
                 "B"_a,
                 "Rin"_a=0.0, "Rout"_a=std::numeric_limits<double>::infinity())
            .def_property_readonly("B", &Plane::getB)
            .def_property_readonly("Rin", &Plane::getRin)
            .def_property_readonly("Rout", &Plane::getRout)
            .def("sag", py::vectorize(&Plane::sag))
            .def("normal", &Plane::normal)
            .def("intercept", &Plane::intercept)
            .def("intersect", (Intersection (Plane::*)(const Ray&) const) &Plane::intersect)
            .def("intersect", (std::vector<Intersection> (Plane::*)(const std::vector<Ray>&) const) &Plane::intersect)
            .def("__repr__", &Plane::repr);
    }
}
