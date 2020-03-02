#include "asphere2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportAsphere2(py::module& m) {
        py::class_<Asphere2, std::shared_ptr<Asphere2>, Surface2>(m, "CPPAsphere2")
            .def(py::init(
                [](
                    double R,
                    double conic,
                    size_t coefptr,
                    size_t size
                ){
                    return new Asphere2(
                        R, conic,
                        reinterpret_cast<double*>(coefptr),
                        size
                    );
                }
            ));
    }
}
