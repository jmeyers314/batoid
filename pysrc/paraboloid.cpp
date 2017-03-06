#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "paraboloid.h"

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
            .def("intersect", (std::vector<Intersection> (Paraboloid::*)(const std::vector<Ray>&) const) &Paraboloid::intersect)
            .def("__repr__", &Paraboloid::repr);
    }
}
