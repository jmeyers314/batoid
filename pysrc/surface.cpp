#include "surface.h"
#include "transformation.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
            .def("intersect", (std::vector<batoid::Intersection> (Surface::*)(const std::vector<batoid::Ray>&) const) &Surface::intersect)
            .def("shift", (Transformation (Surface::*)(double,double,double) const) &Surface::shift)
            .def("shift", (Transformation (Surface::*)(const Vec3&) const) &Surface::shift)
            .def("rotX", &Surface::rotX)
            .def("rotY", &Surface::rotY)
            .def("rotZ", &Surface::rotZ);
    }
}
