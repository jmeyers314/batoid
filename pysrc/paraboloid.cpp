#include "paraboloid.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportParaboloid(py::module& m) {
        py::class_<Paraboloid, std::shared_ptr<Paraboloid>, Surface>(m, "CPPParaboloid")
            .def(py::init<double>(), "init", "R"_a);
    }
}
