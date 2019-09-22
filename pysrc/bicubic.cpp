#include "bicubic.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;
using Eigen::MatrixXd;

namespace batoid {
    void pyExportBicubic(py::module& m) {
        py::class_<Bicubic, std::shared_ptr<Bicubic>, Surface>(m, "CPPBicubic")
            .def(py::init<std::vector<double>, std::vector<double>,
                          const DRef<MatrixXd>, const DRef<MatrixXd>,
                          const DRef<MatrixXd>, const DRef<MatrixXd>, double>(),
                 "init", "xs"_a, "ys"_a, "zs"_a.noconvert(),
                 "dzdxs"_a.noconvert(), "dzdys"_a.noconvert(), "d2zdxdys"_a.noconvert(),
                 "slopFrac"_a
             );
    }
}
