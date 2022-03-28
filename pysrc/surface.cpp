#include "surface.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<SurfaceHandle, std::shared_ptr<SurfaceHandle>>(m, "CPPSurface")
        .def("sag",
            [](const SurfaceHandle& h, size_t xarr, size_t yarr, size_t size, size_t outarr)
            {
                double* xptr = reinterpret_cast<double*>(xarr);
                double* yptr = reinterpret_cast<double*>(yarr);
                double* outptr = reinterpret_cast<double*>(outarr);
                auto s = h.getHostPtr();
                for(int i=0; i<size; i++) {
                    outptr[i] = s->sag(xptr[i], yptr[i]);
                }
            })
        .def("normal",
            [](const SurfaceHandle& h, size_t xarr, size_t yarr, size_t size, size_t outarr)
            {
                double nx, ny, nz;
                double* xptr = reinterpret_cast<double*>(xarr);
                double* yptr = reinterpret_cast<double*>(yarr);
                double* outptr = reinterpret_cast<double*>(outarr);
                auto s = h.getHostPtr();
                for(int i=0; i<size; i++) {
                    s->normal(xptr[i], yptr[i], outptr[i], outptr[i+size], outptr[i+2*size]);
                }
            }
        )
        #if defined(BATOID_GPU)  // For intermediate testing during GPU development
            .def("_sagGPU",
                [](const SurfaceHandle& h, size_t xarr, size_t yarr, size_t size, size_t outarr)
                {
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    double* outptr = reinterpret_cast<double*>(outarr);
                    auto s = h.getPtr();
                    #pragma omp target teams distribute parallel for is_device_ptr(s), \
                            map(to:xptr[:size],yptr[:size]), map(from:outptr[:size])
                    for(size_t i=0; i<size; i++) {
                        outptr[i] = s->sag(xptr[i], yptr[i]);
                    }
                }
            )
        #endif
        ;
    }
}
