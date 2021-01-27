#include "sum.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportSum(py::module& m) {
        py::class_<Sum, std::shared_ptr<Sum>, Surface>(m, "CPPSum")
            .def(py::init(
                [](const std::vector<std::shared_ptr<Surface>>& surfaces) {
                    const Surface** _surfaces = new const Surface*[surfaces.size()];
                    for (int i=0; i<surfaces.size(); i++) {
                        _surfaces[i] = surfaces[i].get();
                    }
                    return new Sum(_surfaces, surfaces.size());
                }
            ));
    }
}
