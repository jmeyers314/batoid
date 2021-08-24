#include "table.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportTable(py::module& m) {
        py::class_<Table, std::shared_ptr<Table>>(m, "CPPTable")
            .def(py::init(
                [](
                    double x0,
                    double y0,
                    double dx,
                    double dy,
                    size_t z_ptr,
                    size_t dzdx_ptr,
                    size_t dzdy_ptr,
                    size_t d2zdxdy_ptr,
                    size_t nx,
                    size_t ny
                ){
                    return new Table(
                        x0, y0, dx, dy,
                        reinterpret_cast<double*>(z_ptr),
                        reinterpret_cast<double*>(dzdx_ptr),
                        reinterpret_cast<double*>(dzdy_ptr),
                        reinterpret_cast<double*>(d2zdxdy_ptr),
                        nx, ny
                    );
                }
            ));
    }
}
