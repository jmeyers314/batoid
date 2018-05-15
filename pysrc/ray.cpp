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
        py::class_<Ray>(m, "Ray")
            .def(py::init<double,double,double,double,double,double,double,double,bool>(),
                 "init",
                 "x0"_a, "y0"_a, "z0"_a, "vx"_a, "vy"_a, "vz"_a, "t"_a=0.0,
                 "w"_a=0.0, "isV"_a=false)
            .def(py::init<Vector3d,Vector3d,double,double,bool>(),
                 "init",
                 "p0"_a, "v"_a, "t"_a=0.0, "w"_a=0.0, "isV"_a=false)
            .def(py::init<bool>(),"init", "failed"_a)
            .def(py::init<Ray>())
            .def_readonly("p0", &Ray::p0)
            .def_readonly("v", &Ray::v)
            .def_readonly("t0", &Ray::t0)
            .def_readonly("wavelength", &Ray::wavelength)
            .def_readonly("isVignetted", &Ray::isVignetted)
            .def_readonly("failed", &Ray::failed)
            .def_property_readonly("x0", [](const Ray& r){ return r.p0[0]; })
            .def_property_readonly("y0", [](const Ray& r){ return r.p0[1]; })
            .def_property_readonly("z0", [](const Ray& r){ return r.p0[2]; })
            .def_property_readonly("vx", [](const Ray& r){ return r.v[0]; })
            .def_property_readonly("vy", [](const Ray& r){ return r.v[1]; })
            .def_property_readonly("vz", [](const Ray& r){ return r.v[2]; })
            .def_property_readonly("k", &Ray::k)
            .def_property_readonly("kx", [](const Ray& r){ return r.k()[0]; })
            .def_property_readonly("ky", [](const Ray& r){ return r.k()[1]; })
            .def_property_readonly("kz", [](const Ray& r){ return r.k()[2]; })
            .def_property_readonly("omega", &Ray::omega)
            .def("positionAtTime", &Ray::positionAtTime)
            .def("__repr__", &Ray::repr)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def("phase", &Ray::phase)
            .def("amplitude", &Ray::amplitude)
            .def("propagatedToTime", &Ray::propagatedToTime)
            .def("propagateInPlace", &Ray::propagateInPlace)
            .def(py::pickle(
                [](const Ray& r) { // __getstate__
                    return py::make_tuple(r.p0, r.v, r.t0, r.wavelength, r.isVignetted, r.failed);
                },
                [](py::tuple t) { // __setstate__
                    Ray r(
                        t[0].cast<Vector3d>(),
                        t[1].cast<Vector3d>(),
                        t[2].cast<double>(),
                        t[3].cast<double>(),
                        t[4].cast<bool>()
                    );
                    if (t[5].cast<bool>())
                        r.setFail();
                    return r;
                }
            ))
            .def("__hash__", [](const Ray& r) {
                auto result = py::hash(py::make_tuple("Ray", r.t0, r.wavelength, r.isVignetted, r.failed));
                const double* d = &r.p0[0];
                for (int i=0; i<3; i++)
                    result = 1000003*result ^ py::hash(py::float_(d[i]));
                d = &r.v[0];
                for (int i=0; i<3; i++)
                    result = 1000003*result ^ py::hash(py::float_(d[i]));
                result = (result == -1) ? -2 : result;
                return result;
            });

    }
}
