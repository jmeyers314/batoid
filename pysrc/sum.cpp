#include "sum.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportSum(py::module& m) {
        py::class_<Sum, std::shared_ptr<Sum>, Surface>(m, "Sum")
            .def(py::init<std::vector<std::shared_ptr<Surface>>>(), "init", "surfaces"_a)
            .def_property_readonly("surfaces", &Sum::getSurfaces)
            .def("__hash__", [](const Sum& s) {
                return py::hash(py::make_tuple(
                    "Sum",
                    py::tuple(py::cast(s.getSurfaces()))
                ));
            });
    }
}
