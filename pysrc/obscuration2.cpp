#include "obscuration2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;


namespace batoid {
    void pyExportObscuration2(py::module& m) {
        py::class_<Obscuration2, std::shared_ptr<Obscuration2>>(m, "CPPObscuration2")
            .def("contains", &Obscuration2::contains)
            .def("obscureInPlace", &Obscuration2::obscureInPlace);

        py::class_<ObscCircle2, std::shared_ptr<ObscCircle2>, Obscuration2>(m, "CPPObscCircle2")
            .def(py::init<double,double,double>(), "init", "radius"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("radius", &ObscCircle2::_radius)
            .def_readonly("x", &ObscCircle2::_x0)
            .def_readonly("y", &ObscCircle2::_y0);

        py::class_<ObscAnnulus2, std::shared_ptr<ObscAnnulus2>, Obscuration2>(m, "CPPObscAnnulus2")
            .def(py::init<double,double,double,double>(), "init", "inner"_a, "outer"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("inner", &ObscAnnulus2::_inner)
            .def_readonly("outer", &ObscAnnulus2::_outer)
            .def_readonly("x", &ObscAnnulus2::_x0)
            .def_readonly("y", &ObscAnnulus2::_y0);

        py::class_<ObscRectangle2, std::shared_ptr<ObscRectangle2>, Obscuration2>(m, "CPPObscRectangle2")
            .def(py::init<double,double,double,double,double>(), "init", "width"_a, "height"_a, "x"_a=0.0, "y"_a=0.0, "th"_a=0.0)
            .def_readonly("width", &ObscRectangle2::_width)
            .def_readonly("height", &ObscRectangle2::_height)
            .def_readonly("x", &ObscRectangle2::_x0)
            .def_readonly("y", &ObscRectangle2::_y0)
            .def_readonly("theta", &ObscRectangle2::_theta);

        py::class_<ObscRay2, std::shared_ptr<ObscRay2>, Obscuration2>(m, "CPPObscRay2")
            .def(py::init<double,double,double,double>(), "init", "width"_a, "theta"_a, "x"_a=0.0, "y"_a=0.0)
            .def_readonly("width", &ObscRay2::_width)
            .def_readonly("theta", &ObscRay2::_theta)
            .def_readonly("x", &ObscRay2::_x0)
            .def_readonly("y", &ObscRay2::_y0);

        py::class_<ObscNegation2, std::shared_ptr<ObscNegation2>, Obscuration2>(m, "CPPObscNegation2")
            .def(py::init<Obscuration2*>(), "init", "original"_a);

        py::class_<ObscUnion2, std::shared_ptr<ObscUnion2>, Obscuration2>(m, "CPPObscUnion2")
            .def(py::init(
                [](const std::vector<std::shared_ptr<Obscuration2>>& obscs) {
                    Obscuration2** _obscs = new Obscuration2*[obscs.size()];
                    for (int i=0; i<obscs.size(); i++) {
                        _obscs[i] = obscs[i].get();
                    }
                    return new ObscUnion2(_obscs, obscs.size());
                }
            ));

        py::class_<ObscIntersection2, std::shared_ptr<ObscIntersection2>, Obscuration2>(m, "CPPObscIntersection2")
            .def(py::init(
                [](const std::vector<std::shared_ptr<Obscuration2>>& obscs) {
                    Obscuration2** _obscs = new Obscuration2*[obscs.size()];
                    for (int i=0; i<obscs.size(); i++) {
                        _obscs[i] = obscs[i].get();
                    }
                    return new ObscIntersection2(_obscs, obscs.size());
                }
            ));
    }
}
