#include <pybind11/pybind11.h>
#include "plane.h"

namespace py = pybind11;

namespace jtrace {
    void pyExportPlane(py::module &m) {
        py::class_<Plane, Surface>(m, "Plane")
            .def(py::init<double>())
            .def_property_readonly("B", &Plane::getB)
            .def("__call__", &Plane::operator())
            .def("normal", &Plane::normal)
            .def("intersect", &Plane::intersect)
            .def("__repr__", &Plane::repr);
    }
}
