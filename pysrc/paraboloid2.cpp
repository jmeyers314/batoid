#include "paraboloid2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportParaboloid2(py::module& m) {
        py::class_<Paraboloid2, std::shared_ptr<Paraboloid2>, Surface2>(m, "CPPParaboloid2")
            .def(py::init<double>(), "init");
    }
}
