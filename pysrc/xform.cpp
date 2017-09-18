#include "xform.h"
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportXForm(py::module& m) {
        py::class_<XForm, std::shared_ptr<XForm>>(m, "XForm")
            .def(py::init<Rot3,Vec3>())
            .def("forward", (Ray (XForm::*)(const Ray&) const) &XForm::to)
            .def("reverse", (Ray (XForm::*)(const Ray&) const) &XForm::from)
            .def("forward", (std::vector<batoid::Ray> (XForm::*)(const std::vector<batoid::Ray>&) const) &XForm::to)
            .def("reverse", (std::vector<batoid::Ray> (XForm::*)(const std::vector<batoid::Ray>&) const) &XForm::from)
            .def("inverse", &XForm::inverse)
            .def(py::self * py::self);
    }
}
