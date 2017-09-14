#include "obscuration.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace batoid {
    void pyExportObscuration(py::module& m) {
        py::class_<Obscuration, std::shared_ptr<Obscuration>>(m, "Obscuration")
            .def("contains", &Obscuration::contains);

        py::class_<ObscCircle, std::shared_ptr<ObscCircle>, Obscuration>(m, "ObscCircle")
            .def(py::init<double,double,double>());

        py::class_<ObscRectangle, std::shared_ptr<ObscRectangle>, Obscuration>(m, "ObscRectangle")
            .def(py::init<double,double,double,double,double>());

        py::class_<ObscRay, std::shared_ptr<ObscRay>, Obscuration>(m, "ObscRay")
            .def(py::init<double,double,double,double>());

        py::class_<ObscUnion, std::shared_ptr<ObscUnion>, Obscuration>(m, "ObscUnion")
            .def(py::init<std::vector<std::shared_ptr<Obscuration>>>());

        py::class_<ObscIntersection, std::shared_ptr<ObscIntersection>, Obscuration>(m, "ObscIntersection")
            .def(py::init<std::vector<std::shared_ptr<Obscuration>>>());

        py::class_<ObscNegation, std::shared_ptr<ObscNegation>, Obscuration>(m, "ObscNegation")
            .def(py::init<std::shared_ptr<Obscuration>>());
    }
}
