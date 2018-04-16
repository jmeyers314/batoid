#include "surface.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
            .def("intercept", (Ray (Surface::*)(const Ray&) const) &Surface::intercept)
            .def("interceptInPlace", (void (Surface::*)(Ray&) const) &Surface::interceptInPlace)
            .def("intercept", (std::vector<Ray> (Surface::*)(const std::vector<Ray>&) const) &Surface::intercept)
            .def("interceptInPlace", (void (Surface::*)(std::vector<Ray>&) const) &Surface::interceptInPlace)
            .def("sag", py::vectorize(&Surface::sag))
            .def("normal", &Surface::normal)
            .def("__repr__", &Surface::repr);
    }
}
