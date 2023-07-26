#include "surface.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "CPPSurface")
            .def("sag", py::vectorize(&Surface::sag))
            .def("normal",
                [](const Surface& s, size_t xarr, size_t yarr, size_t size, size_t outarr)
                {
                    double nx, ny, nz;
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    double* outptr = reinterpret_cast<double*>(outarr);
                    for(int i=0; i<size; i++) {
                        s.normal(xptr[i], yptr[i], outptr[i], outptr[i+size], outptr[i+2*size]);
                    }
                }
            );
    }
}
