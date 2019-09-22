#include "ray.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportRay(py::module& m) {
        py::class_<Ray>(m, "CPPRay")
            .def(py::init<double,double,double,double,double,double,double,double,double,bool>(),
                 "x"_a, "y"_a, "z"_a, "vx"_a, "vy"_a, "vz"_a, "t"_a=0.0,
                 "wavelength"_a=0.0, "flux"_a=1.0, "vignetted"_a=false)
            .def(py::init<Vector3d,Vector3d,double,double,double,bool>(),
                 "r"_a, "v"_a, "t"_a=0.0, "wavelength"_a=0.0, "flux"_a=1.0, "vignetted"_a=false)
            .def(py::init<bool>(), "failed"_a)
            .def(py::init<Ray>())
            .def_readonly("r", &Ray::r)
            .def_readonly("v", &Ray::v)
            .def_readonly("t", &Ray::t)
            .def_readonly("wavelength", &Ray::wavelength)
            .def_readonly("flux", &Ray::flux)
            .def_readonly("vignetted", &Ray::vignetted)
            .def_readonly("failed", &Ray::failed)
            .def_property_readonly("x", [](const Ray& r){ return r.r[0]; })
            .def_property_readonly("y", [](const Ray& r){ return r.r[1]; })
            .def_property_readonly("z", [](const Ray& r){ return r.r[2]; })
            .def_property_readonly("vx", [](const Ray& r){ return r.v[0]; })
            .def_property_readonly("vy", [](const Ray& r){ return r.v[1]; })
            .def_property_readonly("vz", [](const Ray& r){ return r.v[2]; })
            .def_property_readonly("k", &Ray::k)
            .def_property_readonly("kx", [](const Ray& r){ return r.k()[0]; })
            .def_property_readonly("ky", [](const Ray& r){ return r.k()[1]; })
            .def_property_readonly("kz", [](const Ray& r){ return r.k()[2]; })
            .def_property_readonly("omega", &Ray::omega)
            .def("positionAtTime", &Ray::positionAtTime)
            .def("propagatedToTime", &Ray::propagatedToTime)
            .def("propagateInPlace", &Ray::propagateInPlace)
            .def("phase", &Ray::phase)
            .def("amplitude", &Ray::amplitude)
            .def("__repr__", &Ray::repr)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const Ray& r) { // __getstate__
                    return py::make_tuple(
                        r.r, r.v, r.t, r.wavelength, r.flux, r.vignetted, r.failed
                    );
                },
                [](py::tuple t) { // __setstate__
                    Ray r(
                        t[0].cast<Vector3d>(),
                        t[1].cast<Vector3d>(),
                        t[2].cast<double>(),
                        t[3].cast<double>(),
                        t[4].cast<double>(),
                        t[5].cast<bool>()
                    );
                    if (t[6].cast<bool>())
                        r.setFail();
                    return r;
                }
            ))
            .def("__hash__", [](const Ray& r) {
                return py::hash(py::make_tuple(
                    "CPPRay",
                    py::tuple(py::cast(r.r)),
                    py::tuple(py::cast(r.v)),
                    r.t,
                    r.wavelength,
                    r.flux,
                    r.vignetted,
                    r.failed
                ));
            });
    }
}
