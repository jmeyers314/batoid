#include <memory>
#include <pybind11/pybind11.h>
#include "intersection.h"

namespace py = pybind11;

namespace jtrace {
    void pyExportIntersection(py::module& m) {
        py::class_<Intersection, std::shared_ptr<Intersection>>(m, "Intersection")
            .def_readonly("t", &Intersection::t)
            .def_readonly("point", &Intersection::point)
            .def_readonly("surfaceNormal", &Intersection::surfaceNormal)
            .def_property_readonly("x0", &Intersection::getX0)
            .def_property_readonly("y0", &Intersection::getY0)
            .def_property_readonly("z0", &Intersection::getZ0)
            .def_property_readonly("nx", &Intersection::getNx)
            .def_property_readonly("ny", &Intersection::getNy)
            .def_property_readonly("nz", &Intersection::getNz)
            .def("__repr__", &Intersection::repr)
            .def("reflectedRay", &Intersection::reflectedRay)
            .def("refractedRay", &Intersection::refractedRay);
    }
}
