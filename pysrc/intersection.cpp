#include "intersection.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>


PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportIntersection(py::module& m) {
        PYBIND11_NUMPY_DTYPE(Intersection, t, point, surfaceNormal, isVignetted, failed);
        py::class_<Intersection, std::shared_ptr<Intersection>>(m, "Intersection")
            .def(py::init<double,double,double,double,double,double,double,bool>(),
                 "init",
                 "t"_a, "x"_a, "y"_a, "z"_a, "nx"_a, "ny"_a, "nz"_a, "isV"_a=false)
            .def_readonly("t", &Intersection::t)
            .def_readonly("point", &Intersection::point)
            .def_readonly("surfaceNormal", &Intersection::surfaceNormal)
            .def_property_readonly("x0", [](const Intersection& isec){ return isec.point.x; })
            .def_property_readonly("y0", [](const Intersection& isec){ return isec.point.y; })
            .def_property_readonly("z0", [](const Intersection& isec){ return isec.point.z; })
            .def_property_readonly("nx", [](const Intersection& isec){ return isec.surfaceNormal.x; })
            .def_property_readonly("ny", [](const Intersection& isec){ return isec.surfaceNormal.y; })
            .def_property_readonly("nz", [](const Intersection& isec){ return isec.surfaceNormal.z; })
            .def_readonly("isVignetted", &Intersection::isVignetted)
            .def_readonly("failed", &Intersection::failed)
            .def("__repr__", &Intersection::repr)
            .def("reflectedRay", &Intersection::reflectedRay)
            .def("refractedRay", (Ray (Intersection::*)(const Ray&, double, double) const) &Intersection::refractedRay)
            .def("refractedRay", (Ray (Intersection::*)(const Ray&, const Medium&, const Medium&) const) &Intersection::refractedRay)
            .def(py::self == py::self);
        m.def("reflectMany", &reflectMany);
        m.def("refractMany", (std::vector<Ray>(*)(const std::vector<Intersection>&, const std::vector<Ray>&, double, double)) &refractMany);
        m.def("refractMany", (std::vector<Ray>(*)(const std::vector<Intersection>&, const std::vector<Ray>&, const Medium&, const Medium&)) &refractMany);
        auto IV = py::bind_vector<std::vector<Intersection>>(m, "IntersectionVector", py::buffer_protocol());
        IV.def_property_readonly(
            "t",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].t,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "x",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].point.x,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "y",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].point.y,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "z",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].point.z,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "nx",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].surfaceNormal.x,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "ny",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].surfaceNormal.y,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "nz",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<double>(
                    py::buffer_info(
                        &isecs[0].surfaceNormal.z,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "isVignetted",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<bool>(
                    py::buffer_info(
                        &isecs[0].isVignetted,
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        )
        .def_property_readonly(
            "failed",
            [](std::vector<Intersection>& isecs) {
                return py::array_t<bool>(
                    py::buffer_info(
                        &isecs[0].failed,
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        1,
                        {isecs.size()},
                        {sizeof(Intersection)}
                    )
                );
            }
        );
    }
}
