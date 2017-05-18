#include "intersection.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Intersection>);

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
            .def("reflectedRay", (Ray (Intersection::*)(const Ray&) const) &Intersection::reflectedRay)
            .def("reflectedRay", (std::vector<jtrace::Ray> (Intersection::*)(const std::vector<jtrace::Ray>&) const) &Intersection::reflectedRay)
            .def("refractedRay", (Ray (Intersection::*)(const Ray&, double, double) const) &Intersection::refractedRay)
            .def("refractedRay", (std::vector<jtrace::Ray> (Intersection::*)(const std::vector<jtrace::Ray>&, double, double) const) &Intersection::refractedRay)
            .def("refractedRay", (Ray (Intersection::*)(const Ray&, const Medium&, const Medium&) const) &Intersection::refractedRay)
            .def("refractedRay", (std::vector<jtrace::Ray> (Intersection::*)(const std::vector<jtrace::Ray>&, const Medium&, const Medium&) const) &Intersection::refractedRay)
            .def(py::self == py::self);
        py::bind_vector<std::vector<jtrace::Intersection>>(m, "IntersectionVector");
        m.def("reflectMany", &reflectMany);
        m.def("refractMany", (std::vector<jtrace::Ray>(*)(const std::vector<Intersection>&, const std::vector<Ray>&, double, double)) &refractMany);
        m.def("refractMany", (std::vector<jtrace::Ray>(*)(const std::vector<Intersection>&, const std::vector<Ray>&, const Medium&, const Medium&)) &refractMany);
    }
}
