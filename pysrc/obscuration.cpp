#include "obscuration.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportObscuration(py::module& m) {
        py::class_<Obscuration, std::shared_ptr<Obscuration>>(m, "Obscuration")
            .def("contains", &Obscuration::contains)
            .def("obscure", (Ray (Obscuration::*)(const Ray&) const) &Obscuration::obscure)
            .def("obscure", (std::vector<Ray> (Obscuration::*)(const std::vector<Ray>&) const) &Obscuration::obscure);

        py::class_<ObscCircle, std::shared_ptr<ObscCircle>, Obscuration>(m, "ObscCircle")
            .def(py::init<double,double,double>(), "init", "radius"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("radius", &ObscCircle::_radius)
            .def_readonly("x", &ObscCircle::_x0)
            .def_readonly("y", &ObscCircle::_y0);

        py::class_<ObscAnnulus, std::shared_ptr<ObscAnnulus>, Obscuration>(m, "ObscAnnulus")
            .def(py::init<double,double,double,double>(), "init", "inner"_a, "outer"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("inner", &ObscAnnulus::_inner)
            .def_readonly("outer", &ObscAnnulus::_outer)
            .def_readonly("x", &ObscAnnulus::_x0)
            .def_readonly("y", &ObscAnnulus::_y0);

        py::class_<ObscRectangle, std::shared_ptr<ObscRectangle>, Obscuration>(m, "ObscRectangle")
            .def(py::init<double,double,double,double,double>(), "init",
                 "width"_a, "height"_a, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("width", &ObscRectangle::_width)
            .def_readonly("height", &ObscRectangle::_height)
            .def_readonly("theta", &ObscRectangle::_theta)
            .def_readonly("x", &ObscRectangle::_x0)
            .def_readonly("y", &ObscRectangle::_y0);

        py::class_<ObscRay, std::shared_ptr<ObscRay>, Obscuration>(m, "ObscRay")
            .def(py::init<double,double,double,double>(), "init",
                 "width"_a, "theta"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("width", &ObscRay::_width)
            .def_readonly("theta", &ObscRay::_theta)
            .def_readonly("x", &ObscRay::_x0)
            .def_readonly("y", &ObscRay::_y0);

        py::class_<ObscUnion, std::shared_ptr<ObscUnion>, Obscuration>(m, "ObscUnion")
            .def(py::init<std::vector<std::shared_ptr<Obscuration>>>(), "init", "items"_a)
            .def_readonly("items", &ObscUnion::_obscVec);

        py::class_<ObscIntersection, std::shared_ptr<ObscIntersection>, Obscuration>(m, "ObscIntersection")
            .def(py::init<std::vector<std::shared_ptr<Obscuration>>>(), "init", "items"_a)
            .def_readonly("items", &ObscIntersection::_obscVec);

        py::class_<ObscNegation, std::shared_ptr<ObscNegation>, Obscuration>(m, "ObscNegation")
            .def(py::init<std::shared_ptr<Obscuration>>(), "init", "original"_a)
            .def_readonly("original", &ObscNegation::_original);
    }
}
