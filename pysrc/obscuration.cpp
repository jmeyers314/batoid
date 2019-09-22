#include "obscuration.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportObscuration(py::module& m) {
        py::class_<Obscuration, std::shared_ptr<Obscuration>>(m, "CPPObscuration")
            .def("contains", &Obscuration::contains)
            .def("obscure", (Ray (Obscuration::*)(const Ray&) const) &Obscuration::obscure)
            .def("obscure", (RayVector (Obscuration::*)(const RayVector&) const) &Obscuration::obscure)
            .def("obscureInPlace", (void (Obscuration::*)(Ray&) const) &Obscuration::obscureInPlace)
            .def("obscureInPlace", (void (Obscuration::*)(RayVector&) const) &Obscuration::obscureInPlace)
            .def("__repr__", &Obscuration::repr);


        py::class_<ObscCircle, std::shared_ptr<ObscCircle>, Obscuration>(m, "CPPObscCircle")
            .def(py::init<double,double,double>(), "init", "radius"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("radius", &ObscCircle::_radius)
            .def_readonly("x", &ObscCircle::_x0)
            .def_readonly("y", &ObscCircle::_y0)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ObscCircle& oc){ return py::make_tuple(oc._radius, oc._x0, oc._y0); },
                [](py::tuple t) {
                    return ObscCircle(
                        t[0].cast<double>(),
                        t[1].cast<double>(),
                        t[2].cast<double>()
                    );
                }
            ))
            .def("__hash__", [](const ObscCircle& oc) {
                return py::hash(py::make_tuple(
                    "CPPObscCircle",
                    oc._radius,
                    oc._x0,
                    oc._y0
                ));
            });


        py::class_<ObscAnnulus, std::shared_ptr<ObscAnnulus>, Obscuration>(m, "CPPObscAnnulus")
            .def(py::init<double,double,double,double>(), "init", "inner"_a, "outer"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("inner", &ObscAnnulus::_inner)
            .def_readonly("outer", &ObscAnnulus::_outer)
            .def_readonly("x", &ObscAnnulus::_x0)
            .def_readonly("y", &ObscAnnulus::_y0)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ObscAnnulus& oa){ return py::make_tuple(oa._inner, oa._outer, oa._x0, oa._y0); },
                [](py::tuple t) {
                    return ObscAnnulus(
                        t[0].cast<double>(),
                        t[1].cast<double>(),
                        t[2].cast<double>(),
                        t[3].cast<double>()
                    );
                }
            ))
            .def("__hash__", [](const ObscAnnulus& oa) {
                return py::hash(py::make_tuple(
                    "CPPObscAnnulus",
                    oa._inner,
                    oa._outer,
                    oa._x0,
                    oa._y0
                ));
            });


        py::class_<ObscRectangle, std::shared_ptr<ObscRectangle>, Obscuration>(m, "CPPObscRectangle")
            .def(py::init<double,double,double,double,double>(), "init",
                 "width"_a, "height"_a, "x"_a=0.0, "y"_a=0.0, "theta"_a=0.0)
            .def_readonly("width", &ObscRectangle::_width)
            .def_readonly("height", &ObscRectangle::_height)
            .def_readonly("x", &ObscRectangle::_x0)
            .def_readonly("y", &ObscRectangle::_y0)
            .def_readonly("theta", &ObscRectangle::_theta)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ObscRectangle& o){ return py::make_tuple(o._width, o._height, o._x0, o._y0, o._theta); },
                [](py::tuple t) {
                    return ObscRectangle(
                        t[0].cast<double>(),
                        t[1].cast<double>(),
                        t[2].cast<double>(),
                        t[3].cast<double>(),
                        t[4].cast<double>()
                    );
                }
            ))
            .def("__hash__", [](const ObscRectangle& o) {
                return py::hash(py::make_tuple(
                    "CPPObscRectangle",
                    o._width,
                    o._height,
                    o._x0,
                    o._y0,
                    o._theta
                ));
            });


        py::class_<ObscRay, std::shared_ptr<ObscRay>, Obscuration>(m, "CPPObscRay")
            .def(py::init<double,double,double,double>(), "init",
                 "width"_a, "theta"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("width", &ObscRay::_width)
            .def_readonly("theta", &ObscRay::_theta)
            .def_readonly("x", &ObscRay::_x0)
            .def_readonly("y", &ObscRay::_y0)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ObscRay& o){ return py::make_tuple(o._width, o._theta, o._x0, o._y0); },
                [](py::tuple t) {
                    return ObscRay(
                        t[0].cast<double>(),
                        t[1].cast<double>(),
                        t[2].cast<double>(),
                        t[3].cast<double>()
                    );
                }
            ))
            .def("__hash__", [](const ObscRay& o) {
                return py::hash(py::make_tuple(
                    "CPPObscRay",
                    o._width,
                    o._theta,
                    o._x0,
                    o._y0
                ));
            });


        py::class_<ObscNegation, std::shared_ptr<ObscNegation>, Obscuration>(m, "CPPObscNegation")
            .def(py::init<std::shared_ptr<Obscuration>>(), "init", "original"_a)
            .def_readonly("original", &ObscNegation::_original)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ObscNegation& o){ return o._original; },
                [](std::shared_ptr<Obscuration>& o) { return ObscNegation(o); }
            ))
            .def("__hash__", [](const ObscNegation& o) {
                return py::hash(py::make_tuple(
                    "CPPObscNegation",
                    o._original
                ));
            });


        py::class_<ObscUnion, std::shared_ptr<ObscUnion>, Obscuration>(m, "CPPObscUnion")
            .def(py::init<std::vector<std::shared_ptr<Obscuration>>>(), "init", "items"_a)
            .def_readonly("items", &ObscUnion::_obscVec)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ObscUnion& o){ return o._obscVec; },
                [](std::vector<std::shared_ptr<Obscuration>>& o) { return ObscUnion(o); }
            ))
            .def("__hash__", [](const ObscUnion& o) {
                return py::hash(py::make_tuple(
                    "CPPObscUnion",
                    py::tuple(py::cast(o._obscVec))
                ));
            });


        py::class_<ObscIntersection, std::shared_ptr<ObscIntersection>, Obscuration>(m, "CPPObscIntersection")
            .def(py::init<std::vector<std::shared_ptr<Obscuration>>>(), "init", "items"_a)
            .def_readonly("items", &ObscIntersection::_obscVec)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ObscIntersection& o){ return o._obscVec; },
                [](std::vector<std::shared_ptr<Obscuration>>& o) { return ObscIntersection(o); }
            ))
            .def("__hash__", [](const ObscIntersection& o) {
                return py::hash(py::make_tuple(
                    "CPPObscIntersection",
                    py::tuple(py::cast(o._obscVec))
                ));
            });
    }
}
