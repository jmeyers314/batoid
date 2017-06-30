#include "intersection.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>


PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportIntersection(py::module& m) {
        py::class_<Intersection, std::shared_ptr<Intersection>>(m, "Intersection")
            .def(py::init<double,double,double,double,double,double,double,bool>(),
                 "init",
                 "t"_a, "x"_a, "y"_a, "z"_a, "nx"_a, "ny"_a, "nz"_a, "isV"_a=false)
            .def_readonly("t", &Intersection::t)
            .def_readonly("point", &Intersection::point)
            .def_readonly("surfaceNormal", &Intersection::surfaceNormal)
            .def_property_readonly("x0", &Intersection::getX0)
            .def_property_readonly("y0", &Intersection::getY0)
            .def_property_readonly("z0", &Intersection::getZ0)
            .def_property_readonly("nx", &Intersection::getNx)
            .def_property_readonly("ny", &Intersection::getNy)
            .def_property_readonly("nz", &Intersection::getNz)
            .def_readonly("isVignetted", &Intersection::isVignetted)
            .def_readonly("failed", &Intersection::failed)
            .def("__repr__", &Intersection::repr)
            .def("reflectedRay", (Ray (Intersection::*)(const Ray&) const) &Intersection::reflectedRay)
            .def("reflectedRay", (std::vector<jtrace::Ray> (Intersection::*)(const std::vector<jtrace::Ray>&) const) &Intersection::reflectedRay)
            .def("refractedRay", (Ray (Intersection::*)(const Ray&, double, double) const) &Intersection::refractedRay)
            .def("refractedRay", (std::vector<jtrace::Ray> (Intersection::*)(const std::vector<jtrace::Ray>&, double, double) const) &Intersection::refractedRay)
            .def("refractedRay", (Ray (Intersection::*)(const Ray&, const Medium&, const Medium&) const) &Intersection::refractedRay)
            .def("refractedRay", (std::vector<jtrace::Ray> (Intersection::*)(const std::vector<jtrace::Ray>&, const Medium&, const Medium&) const) &Intersection::refractedRay)
            .def(py::self == py::self);
        m.def("reflectMany", &reflectMany);
        m.def("refractMany", (std::vector<jtrace::Ray>(*)(const std::vector<Intersection>&, const std::vector<Ray>&, double, double)) &refractMany);
        m.def("refractMany", (std::vector<jtrace::Ray>(*)(const std::vector<Intersection>&, const std::vector<Ray>&, const Medium&, const Medium&)) &refractMany);
        auto IV = py::bind_vector<std::vector<jtrace::Intersection>>(m, "IntersectionVector");
        IV.def_property_readonly(
            "t",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].t,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "x",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].point.x,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "y",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].point.y,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "z",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].point.z,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "nx",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].surfaceNormal.x,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "ny",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].surfaceNormal.y,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "nz",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].surfaceNormal.z,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "isVignetted",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<bool>(
                    py::buffer_info(
                        &isecs[0].isVignetted,
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "failed",
            [](std::vector<jtrace::Intersection>& isecs) {
                return py::array_t<bool>(
                    py::buffer_info(
                        &isecs[0].failed,
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(jtrace::Intersection)}
                    )
                );
            }
        );
    }
}
