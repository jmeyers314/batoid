#include "surface.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def("sag", py::vectorize(&Surface::sag))
            .def("normal", &Surface::normal)

            .def("intersect", (Ray (Surface::*)(const Ray&) const) &Surface::intersect)
            .def("intersect", (RayVector (Surface::*)(const RayVector&) const) &Surface::intersect)
            .def("intersectInPlace", (void (Surface::*)(Ray&) const) &Surface::intersectInPlace)
            .def("intersectInPlace", (void (Surface::*)(RayVector&) const) &Surface::intersectInPlace)

            .def("reflect", (Ray (Surface::*)(const Ray&) const) &Surface::reflect)
            .def("reflect", (RayVector (Surface::*)(const RayVector&) const) &Surface::reflect)
            .def("reflectInPlace", (void (Surface::*)(Ray&) const) &Surface::reflectInPlace)
            .def("reflectInPlace", (void (Surface::*)(RayVector&) const) &Surface::reflectInPlace)

            .def("refract", (Ray (Surface::*)(const Ray&, const Medium&, const Medium&) const) &Surface::refract)
            .def("refract", (RayVector (Surface::*)(const RayVector&, const Medium&, const Medium&) const) &Surface::refract)
            .def("refractInPlace", (void (Surface::*)(Ray&, const Medium&, const Medium&) const) &Surface::refractInPlace)
            .def("refractInPlace", (void (Surface::*)(RayVector&, const Medium&, const Medium&) const) &Surface::refractInPlace);
    }
}
