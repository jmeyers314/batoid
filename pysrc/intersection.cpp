#include <pybind11/pybind11.h>
#include "intersection.h"

namespace py = pybind11;

namespace jtrace {
    void pyExportIntersection(py::module &m) {
        py::class_<Intersection>(m, "Intersection")
            .def_readonly("t", &Intersection::t)
            .def_readonly("point", &Intersection::point)
            .def_readonly("surfaceNormal", &Intersection::surfaceNormal)
            .def("__repr__", &Intersection::repr)
            .def("reflectedRay", &Intersection::reflectedRay);
    }
}
