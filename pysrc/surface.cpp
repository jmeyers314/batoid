#include "surface.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace batoid {
    void pyExportSurface(py::module& m) {
        py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
            .def("sag", py::vectorize(&Surface::sag))
            .def("normal", (Vector3d (Surface::*)(double, double) const) &Surface::normal)
            .def("normal",
                [](const Surface& s, py::array_t<double> xs, py::array_t<double> ys) -> py::array_t<double>
                {
                    auto bufX = xs.request();
                    auto bufY = ys.request();
                    if (bufX.ndim != bufY.ndim)
                        throw std::runtime_error("Dimensions must match");
                    if (bufX.size != bufY.size)
                        throw std::runtime_error("Sizes much match");

                    // Create output as a std::vector and fill it in
                    std::vector<Vector3d> result;
                    result.reserve(bufX.size);
                    std::vector<unsigned int> idxVec(bufX.ndim, 0);
                    for (ssize_t idx=0; idx<bufX.size; idx++) {
                        char *ptrX = (char *) bufX.ptr;
                        char *ptrY = (char *) bufY.ptr;
                        for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                            ptrX += idxVec[idim]*bufX.strides[idim];
                            ptrY += idxVec[idim]*bufY.strides[idim];
                        }
                        result.push_back(s.normal(*(double *)ptrX, *(double *)ptrY));
                        for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                            idxVec[idim]++;
                            if (idxVec[idim] == bufX.shape[idim])
                                idxVec[idim] = 0;
                            else
                                break;
                        }
                    }

                    std::vector<long> newShape(bufX.shape);
                    newShape.push_back(3l);

                    std::vector<long> newStrides;
                    long val = sizeof(double);
                    newStrides.push_back(val);
                    for (ssize_t i=newShape.size()-1; i>0; i--) {
                        val *= newShape[i];
                        newStrides.push_back(val);
                    }
                    std::reverse(std::begin(newStrides), std::end(newStrides));

                    return py::array_t<double>(
                        newShape, newStrides,
                        &result[0].data()[0]
                        );
                }
            )

            .def("intersect", &Surface::intersect, py::arg(), py::arg()=nullptr)
            .def("intersectInPlace", &Surface::intersectInPlace, py::arg(), py::arg()=nullptr)
            .def("reflect", &Surface::reflect, py::arg(), py::arg()=nullptr, py::arg()=nullptr)
            .def("reflectInPlace", &Surface::reflectInPlace, py::arg(), py::arg()=nullptr, py::arg()=nullptr)

            .def("refract", &Surface::refract, py::arg(), py::arg(), py::arg(), py::arg()=nullptr, py::arg()=nullptr)
            .def("refractInPlace", &Surface::refractInPlace, py::arg(), py::arg(), py::arg(), py::arg()=nullptr, py::arg()=nullptr)

            .def("rSplit", &Surface::rSplit, py::arg(), py::arg(), py::arg(), py::arg(), py::arg()=nullptr);
            // .def("rSplitProb", &Surface::rSplitProb);
    }
}
