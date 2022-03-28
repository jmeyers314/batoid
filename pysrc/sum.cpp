#include "sum.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportSum(py::module& m) {
        py::class_<SumHandle, std::shared_ptr<SumHandle>, SurfaceHandle>(m, "CPPSum")
            .def(py::init(
                [](const std::vector<std::shared_ptr<SurfaceHandle>>& handles) {
                    const SurfaceHandle** _handles = new const SurfaceHandle*[handles.size()];
                    for (int i=0; i<handles.size(); i++) {
                        _handles[i] = handles[i].get();
                    }
                    return new SumHandle(_handles, handles.size());
                }
            ));
    }
}
