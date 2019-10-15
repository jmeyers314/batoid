#include "rayVector2.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace batoid {
    void pyExportRayVector2(py::module& m) {
        auto rv2 = py::class_<RayVector2>(m, "CPPRayVector2")
            .def(py::init<Ref<MatrixX3d>, Ref<MatrixX3d>, Ref<VectorXd>, Ref<VectorXd>, Ref<VectorXd>, Ref<VectorXb>, Ref<VectorXb>>())
            .def("synchronize", &RayVector2::synchronize)
            .def("sendToDevice", &RayVector2::sendToDevice)
  	    .def("inspect", &RayVector2::inspect)
  	    .def_property_readonly("owner", [](const RayVector2& rv){return rv.owner;});
	py::enum_<RayVector2::OwnerType>(rv2, "Owner")
	  .value("host", RayVector2::OwnerType::host)
	  .value("device", RayVector2::OwnerType::device);
    }
}
