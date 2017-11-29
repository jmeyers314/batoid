#include "surface.h"
#include "transformation.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
            .def("intersect", (Intersection (Surface::*)(const Ray&) const) &Surface::intersect)
            .def("intercept", (Ray (Surface::*)(const Ray&) const) &Surface::intercept)
            .def("interceptInPlace", (void (Surface::*)(Ray&) const) &Surface::interceptInPlace)
            .def("intersect", (std::vector<Intersection> (Surface::*)(const std::vector<Ray>&) const) &Surface::intersect)
            .def("intercept", (std::vector<Ray> (Surface::*)(const std::vector<Ray>&) const) &Surface::intercept)
            .def("interceptInPlace", (void (Surface::*)(std::vector<Ray>&) const) &Surface::interceptInPlace)
            .def("sag", py::vectorize(&Surface::sag))
            .def("normal", &Surface::normal)
            .def("__repr__", &Surface::repr)
            .def("shift", (Transformation (Surface::*)(double,double,double) const) &Surface::shift)
            .def("shift", (Transformation (Surface::*)(const Vec3&) const) &Surface::shift)
            .def("rotX", &Surface::rotX)
            .def("rotY", &Surface::rotY)
            .def("rotZ", &Surface::rotZ);
    }
}
