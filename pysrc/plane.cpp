#include "plane.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Intersection>);

namespace py = pybind11;

namespace jtrace {
    void pyExportPlane(py::module& m) {
        py::class_<Plane, std::shared_ptr<Plane>, Surface>(m, "Plane")
            .def(py::init<double>())
            .def_property_readonly("B", &Plane::getB)
            .def("__call__", &Plane::operator())
            .def("normal", &Plane::normal)
            .def("intersect", (Intersection (Plane::*)(const Ray&) const) &Plane::intersect)
            .def("intersect", (std::vector<jtrace::Intersection> (Plane::*)(const std::vector<jtrace::Ray>&) const) &Plane::intersect)
            .def("__repr__", &Plane::repr);
    }
}
