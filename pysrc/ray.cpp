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

        // Collect documentation first, then define the class below.

        std::string docRay(R"pbdoc(
            Ray
            A Ray can alternately be thought of as a geometric ray or as a monochromatic plane wave.
        )pbdoc");

        std::string docRayP0(R"pbdoc(
            (3,), ndarray of float
            Reference point in meters.
        )pbdoc");

        std::string docRayV(R"pbdoc(
            (3,), ndarray of float
            Velocity vector in units of the speed of light.  Note this may have magnitude < 1
            if the Ray is inside a refractive medium.
        )pbdoc");

        std::string docRayT0(R"pbdoc(
            float
            Reference time (over the speed of light) in units of meters.
        )pbdoc");

        std::string docRayWavelength(R"pbdoc(
            float
            Vacuum wavelength in meters.
        )pbdoc");

        std::string docRayIsVignetted(R"pbdoc(
            bool
            Whether Ray has been vignetted or not.
        )pbdoc");

        std::string docRayFailed(R"pbdoc(
            bool
            Whether Ray is in failed state or not, which may happen if an intersection with a
            surface is requested but cannot be found.
        )pbdoc");

        std::string docRayX0(R"pbdoc(
            float
            X-coordinate of reference position in meters.
        )pbdoc");
        std::string docRayY0(R"pbdoc(
            float
            Y-coordinate of reference position in meters.
        )pbdoc");
        std::string docRayZ0(R"pbdoc(
            float
            Z-coordinate of reference position in meters.
        )pbdoc");

        std::string docRayVX(R"pbdoc(
            float
            X-coordinate of velocity vector in units of the speed of light.
        )pbdoc");
        std::string docRayVY(R"pbdoc(
            float
            Y-coordinate of velocity vector in units of the speed of light.
        )pbdoc");
        std::string docRayVZ(R"pbdoc(
            float
            Z-coordinate of velocity vector in units of the speed of light.
        )pbdoc");

        std::string docRayK(R"pbdoc(
            (3,) ndarray of float
            Wavevector of planewave in units of radians per meter.  The magnitude of the wavevector
            is equal to $2 \pi n / \lambda$, where $n$ is the refractive index and $\lambda$ is the
            wavelength.
        )pbdoc");

        std::string docRayKX(R"pbdoc(
            float
            X-coordinate of wavevector in units of radians per meter.
        )pbdoc");
        std::string docRayKY(R"pbdoc(
            float
            Y-coordinate of wavevector in units of radians per meter.
        )pbdoc");
        std::string docRayKZ(R"pbdoc(
            float
            Z-coordinate of wavevector in units of radians per meter.
        )pbdoc");

        std::string docRayOmega(R"pbdoc(
            float
            Temporal frequency of the planewave over the speed of light.  Units are inverse meters.
            Equals $2 \pi / \lambda$.
        )pbdoc");

        std::string docRayPositionAtTime(R"pbdoc(
            Calculate the position of the Ray at a given time.

            Parameters
            ----------
            t : float
                Time (over the speed of light; in meters) at which to compute position.

            Returns
            -------
            position : (3,), ndarray of float
               Position in meters.
        )pbdoc");

        std::string docRayPropagatedToTime(R"pbdoc(
            Return a Ray propagated to given time.

            Parameters
            ----------
            t : float
                Time (over the speed of light; in meters) to which to propagate ray.

            Returns
            -------
            Ray
        )pbdoc");

        std::string docRayPropagateInPlace(R"pbdoc(
            Propagate Ray to given time.

            Parameters
            ----------
            t : float
                Time (over the speed of light; in meters) to which to propagate ray.
        )pbdoc");

        std::string docRayPhase(R"pbdoc(
            Calculate plane wave phase at given position and time.

            Parameters
            ----------
            position : (3,), ndarray of float
                Position at which to compute phase
            time : float
                Time (over the speed of light; in meters) at which to compute phase

            Returns
            -------
            phase : float
        )pbdoc");

        std::string docRayAmplitude(R"pbdoc(
            Calculate (scalar) complex electric-field amplitude at given position and time.

            Parameters
            ----------
            position : (3,), ndarray of float
                Position in meters at which to compute phase
            time : float
                Time (over the speed of light; in meters) at which to compute phase

            Returns
            -------
            amplitude : complex
        )pbdoc");

        std::string docRayInit1(R"pbdoc(
            Parameters
            ----------
            x0, y0, z0 : float
                Reference position (meters)
            t0 : float
                Reference time over the speed of light; in meters.
        )pbdoc");

        py::options options;
        options.disable_function_signatures();

        py::class_<Ray>(m, "Ray", docRay.c_str())
            .def(py::init<double,double,double,double,double,double,double,double,bool>(),
                 "x0"_a, "y0"_a, "z0"_a, "vx"_a, "vy"_a, "vz"_a, "t"_a=0.0,
                 "wavelength"_a=0.0, "isVignetted"_a=false,
                 "banana doc")
            .def(py::init<Vector3d,Vector3d,double,double,bool>(),
                 "p0"_a, "v"_a, "t"_a=0.0, "wavelength"_a=0.0, "isVignetted"_a=false,
                 "papaya doc")
            .def(py::init<bool>(), "failed"_a)
            .def(py::init<Ray>())
            .def_readonly("p0", &Ray::p0, docRayP0.c_str())
            .def_readonly("v", &Ray::v, docRayV.c_str())
            .def_readonly("t0", &Ray::t0, docRayT0.c_str())
            .def_readonly("wavelength", &Ray::wavelength, docRayWavelength.c_str())
            .def_readonly("isVignetted", &Ray::isVignetted, docRayIsVignetted.c_str())
            .def_readonly("failed", &Ray::failed, docRayFailed.c_str())
            .def_property_readonly("x0", [](const Ray& r){ return r.p0[0]; }, docRayX0.c_str())
            .def_property_readonly("y0", [](const Ray& r){ return r.p0[1]; }, docRayY0.c_str())
            .def_property_readonly("z0", [](const Ray& r){ return r.p0[2]; }, docRayZ0.c_str())
            .def_property_readonly("vx", [](const Ray& r){ return r.v[0]; }, docRayVX.c_str())
            .def_property_readonly("vy", [](const Ray& r){ return r.v[1]; }, docRayVY.c_str())
            .def_property_readonly("vz", [](const Ray& r){ return r.v[2]; }, docRayVZ.c_str())
            .def_property_readonly("k", &Ray::k, docRayK.c_str())
            .def_property_readonly("kx", [](const Ray& r){ return r.k()[0]; }, docRayKX.c_str())
            .def_property_readonly("ky", [](const Ray& r){ return r.k()[1]; }, docRayKY.c_str())
            .def_property_readonly("kz", [](const Ray& r){ return r.k()[2]; }, docRayKZ.c_str())
            .def_property_readonly("omega", &Ray::omega, docRayOmega.c_str())
            .def("positionAtTime", &Ray::positionAtTime, docRayPositionAtTime.c_str())
            .def("propagatedToTime", &Ray::propagatedToTime, docRayPropagatedToTime.c_str())
            .def("propagateInPlace", &Ray::propagateInPlace, docRayPropagateInPlace.c_str())
            .def("phase", &Ray::phase, docRayPhase.c_str())
            .def("amplitude", &Ray::amplitude, docRayAmplitude.c_str())

            .def("__repr__", &Ray::repr)
            .def(py::self == py::self)
            .def(py::self != py::self)

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
                return py::hash(py::make_tuple(
                    "Ray",
                    py::tuple(py::cast(r.p0)),
                    py::tuple(py::cast(r.v)),
                    r.t0,
                    r.wavelength,
                    r.isVignetted,
                    r.failed
                ));
            });

    }
}
