#include "rayVector3.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

namespace py = pybind11;

namespace batoid {
    void pyExportRayVector3(py::module& m) {
        auto rv3 = py::class_<RayVector3>(m, "CPPRayVector3")
            .def(py::init<Ref<MatrixX3d>, Ref<MatrixX3d>, Ref<VectorXd>, Ref<VectorXd>, Ref<VectorXd>, Ref<VectorXb>, Ref<VectorXb>>())
            .def("synchronize", &RayVector2::synchronize)
            .def("sendToDevice", &RayVector2::sendToDevice)
  	    .def_property_readonly("owner", [](const RayVector2& rv){return rv.owner;})
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("positionAtTime", py::overload_cast<double, Ref<MatrixX3d>>(&RayVector2::positionAtTime, py::const_));

        py::enum_<RayVector2::OwnerType>(rv2, "Owner")
    	  .value("host", RayVector2::OwnerType::host)
    	  .value("device", RayVector2::OwnerType::device);

        m.def("runtime_init", &runtime_init);
    }
}
