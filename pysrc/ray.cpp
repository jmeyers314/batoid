#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "ray.h"

PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Ray>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportRay(py::module& m) {
        py::class_<Ray>(m, "Ray")
            .def(py::init<double,double,double,double,double,double,double>(),
                 "init",
                 "x0"_a, "y0"_a, "z0"_a, "vx"_a, "vy"_a, "vz"_a, "t"_a=0.0)
            .def(py::init<Vec3,Vec3,double>(),
                 "init",
                 "p0"_a, "v"_a, "t"_a=0.0)
            .def(py::init<std::array<double,3>,std::array<double,3>,double>())
            .def_readonly("p0", &Ray::p0)
            .def_readonly("v", &Ray::v)
            .def_readonly("t0", &Ray::t0)
            .def_property_readonly("x0", &Ray::getX0)
            .def_property_readonly("y0", &Ray::getY0)
            .def_property_readonly("z0", &Ray::getZ0)
            .def_property_readonly("vx", &Ray::getVx)
            .def_property_readonly("vy", &Ray::getVy)
            .def_property_readonly("vz", &Ray::getVz)
            .def("__call__", &Ray::operator())
            .def("__repr__", &Ray::repr);
        py::bind_vector<std::vector<jtrace::Ray>>(m, "RayVector");
    }
}
