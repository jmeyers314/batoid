#include "ray.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>


PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Ray>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportRay(py::module& m) {
        PYBIND11_NUMPY_DTYPE(Ray, p0, v, t0, wavelength, isVignetted, failed);
        py::class_<Ray>(m, "Ray")
            .def(py::init<double,double,double,double,double,double,double,double,bool>(),
                 "init",
                 "x0"_a, "y0"_a, "z0"_a, "vx"_a, "vy"_a, "vz"_a, "t"_a=0.0,
                 "w"_a=0.0, "isV"_a=false)
            .def(py::init<Vec3,Vec3,double,double,bool>(),
                 "init",
                 "p0"_a, "v"_a, "t"_a=0.0, "w"_a=0.0, "isV"_a=false)
            .def(py::init<std::array<double,3>,std::array<double,3>,double,double,bool>(),
                 "init",
                 "p0"_a, "v"_a, "t"_a=0.0, "w"_a=0.0, "isV"_a=false)
            .def_readonly("p0", &Ray::p0)
            .def_readonly("v", &Ray::v)
            .def_readonly("t0", &Ray::t0)
            .def_readonly("wavelength", &Ray::wavelength)
            .def_readonly("isVignetted", &Ray::isVignetted)
            .def_readonly("failed", &Ray::failed)
            .def_property_readonly("x0", &Ray::getX0)
            .def_property_readonly("y0", &Ray::getY0)
            .def_property_readonly("z0", &Ray::getZ0)
            .def_property_readonly("vx", &Ray::getVx)
            .def_property_readonly("vy", &Ray::getVy)
            .def_property_readonly("vz", &Ray::getVz)
            .def_property_readonly("k", &Ray::k)
            .def_property_readonly("omega", &Ray::omega)
            .def("positionAtTime", &Ray::positionAtTime)
            .def("__repr__", &Ray::repr)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def("phase", &Ray::phase)
            .def("amplitude", &Ray::amplitude)
            .def("propagatedToTime", &Ray::propagatedToTime);
        m.def("amplitudeMany", &amplitudeMany);
        m.def("phaseMany", &phaseMany);
        m.def("propagatedToTimesMany", &propagatedToTimesMany);

        auto RV = py::bind_vector<std::vector<jtrace::Ray>>(m, "RayVector", py::buffer_protocol());
        // This feels a little hacky, but seems to work.
        RV.def_property_readonly(
            "x",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].p0.x,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "y",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].p0.y,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "z",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].p0.z,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "vx",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].v.x,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "vy",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].v.y,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "vz",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].v.z,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "t0",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].t0,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "wavelength",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<double>(
                    py::buffer_info(
                        &rv[0].wavelength,
                        sizeof(double),
                        py::format_descriptor<double>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "isVignetted",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<bool>(
                    py::buffer_info(
                        &rv[0].isVignetted,
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        )
        .def_property_readonly(
            "failed",
            [](std::vector<jtrace::Ray>& rv) {
                return py::array_t<bool>(
                    py::buffer_info(
                        &rv[0].failed,
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        1,
                        {rv.size()},
                        {sizeof(jtrace::Ray)}
                    )
                );
            }
        );
    }
}
