#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include "paraboloid.h"

PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Intersection>);

namespace py = pybind11;

namespace jtrace {
    void pyExportParaboloid(py::module& m) {
        py::class_<Paraboloid, std::shared_ptr<Paraboloid>, Surface>(m, "Paraboloid")
            .def(py::init<double,double>())
            .def_property_readonly("A", &Paraboloid::getA)
            .def_property_readonly("B", &Paraboloid::getB)
            .def("__call__", &Paraboloid::operator())
            .def("normal", &Paraboloid::normal)
            .def("intersect", (Intersection (Paraboloid::*)(const Ray&) const) &Paraboloid::intersect)
            .def("intersect", (std::vector<jtrace::Intersection> (Paraboloid::*)(const std::vector<jtrace::Ray>&) const) &Paraboloid::intersect)
            .def("__repr__", &Paraboloid::repr);
    }
}
