#include <pybind11/pybind11.h>
#include "paraboloid.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportParaboloid(py::module &m) {
        py::class_<Paraboloid, Surface>(m, "Paraboloid")
            .def(py::init<double,double>())
            .def_property_readonly("A", &Paraboloid::getA)
            .def_property_readonly("B", &Paraboloid::getB)
            .def("intersect", &Paraboloid::intersect)
            .def("__call__", &Paraboloid::operator())
            .def("__repr__", &Paraboloid::repr);
    }
}
