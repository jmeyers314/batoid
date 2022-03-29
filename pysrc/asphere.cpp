#include "asphere.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportAsphere(py::module& m) {
        py::class_<AsphereHandle, std::shared_ptr<AsphereHandle>, SurfaceHandle>(m, "CPPAsphere")
            .def(py::init(
                [](
                    double R,
                    double conic,
                    size_t coefptr,
                    size_t size
                ){
                    return new AsphereHandle(
                        R, conic,
                        reinterpret_cast<double*>(coefptr),
                        size
                    );
                }
            ));
    }
}
