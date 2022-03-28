#include "tilted.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportTilted(py::module& m) {
        py::class_<TiltedHandle, std::shared_ptr<TiltedHandle>, SurfaceHandle>(m, "CPPTilted")
            .def(py::init<double,double>(), "init", "tanx"_a, "tany"_a);
    }

}
