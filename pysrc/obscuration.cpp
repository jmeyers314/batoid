#include "obscuration.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportObscuration(py::module& m) {
        py::class_<Obscuration, std::shared_ptr<Obscuration>>(m, "CPPObscuration")
            .def("contains", py::vectorize(&Obscuration::contains));


        py::class_<ObscCircle, std::shared_ptr<ObscCircle>, Obscuration>(m, "CPPObscCircle")
            .def(py::init<double,double,double>(), "init", "radius"_a, "x"_a=0.0, "y"_a=0.0);


        py::class_<ObscAnnulus, std::shared_ptr<ObscAnnulus>, Obscuration>(m, "CPPObscAnnulus")
            .def(py::init<double,double,double,double>(), "init", "inner"_a, "outer"_a, "x"_a=0.0, "y"_a=0.0);


        // py::class_<ObscRectangle, std::shared_ptr<ObscRectangle>, Obscuration>(m, "CPPObscRectangle")
        //     .def(py::init<double,double,double,double,double>(), "init",
        //          "width"_a, "height"_a, "x"_a=0.0, "y"_a=0.0, "theta"_a=0.0)
        //
        //
        // py::class_<ObscRay, std::shared_ptr<ObscRay>, Obscuration>(m, "CPPObscRay")
        //     .def(py::init<double,double,double,double>(), "init",
        //          "width"_a, "theta"_a, "x"_a=0.0, "y"_a=0.0)
        //
        //
        // py::class_<ObscPolygon, std::shared_ptr<ObscPolygon>, Obscuration>(m, "CPPObscPolygon")
        //     .def(py::init<std::vector<double>, std::vector<double>>(), "init")
        //
        //
        // py::class_<ObscNegation, std::shared_ptr<ObscNegation>, Obscuration>(m, "CPPObscNegation")
        //     .def(py::init<std::shared_ptr<Obscuration>>(), "init", "original"_a)
        //
        //
        // py::class_<ObscUnion, std::shared_ptr<ObscUnion>, Obscuration>(m, "CPPObscUnion")
        //     .def(py::init<std::vector<std::shared_ptr<Obscuration>>>(), "init", "items"_a)
        //
        //
        // py::class_<ObscIntersection, std::shared_ptr<ObscIntersection>, Obscuration>(m, "CPPObscIntersection")
        //     .def(py::init<std::vector<std::shared_ptr<Obscuration>>>(), "init", "items"_a)
    }
}
