#include "surface.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
            .def("intersect", (Ray (Surface::*)(const Ray&) const) &Surface::intersect)
            .def("intersectInPlace", (void (Surface::*)(Ray&) const) &Surface::intersectInPlace)
            .def("intersect",
                [](const Surface& s, const RayVector& rv) {
                    RayVector result;
                    result.rays = std::move(s.intersect(rv.rays));
                    return result;
                }
            )
            .def("intersectInPlace", [](const Surface& s, RayVector& rv){
                s.intersectInPlace(rv.rays);
            })
            .def("sag", py::vectorize(&Surface::sag))
            .def("normal", &Surface::normal)
            .def("__repr__", &Surface::repr);
    }
}
