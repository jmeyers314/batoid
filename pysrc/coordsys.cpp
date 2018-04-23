#include "coordsys.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <tuple>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);

namespace py = pybind11;

namespace batoid {
    void pyExportCoordSys(py::module& m) {
        py::class_<CoordSys, std::shared_ptr<CoordSys>>(m, "CoordSys")
            .def(py::init<>())
            .def(py::init<Vector3d>())
            .def(py::init<Matrix3d>())
            .def(py::init<Vector3d,Matrix3d>())
            .def_readonly("origin", &CoordSys::m_origin, "Global origin")
            .def_readonly("rot", &CoordSys::m_rot, "Unit vector rotation matrix")
            .def_property_readonly("xhat", &CoordSys::getXHat)
            .def_property_readonly("yhat", &CoordSys::getYHat)
            .def_property_readonly("zhat", &CoordSys::getZHat)
            .def("__repr__", &CoordSys::repr)
            .def("shiftGlobal", &CoordSys::shiftGlobal)
            .def("shiftLocal", &CoordSys::shiftLocal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Matrix3d&) const) &CoordSys::rotateGlobal)
            .def("rotateGlobal", (CoordSys (CoordSys::*) (const Matrix3d&, const Vector3d&, const CoordSys&) const) &CoordSys::rotateGlobal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Matrix3d&) const) &CoordSys::rotateLocal)
            .def("rotateLocal", (CoordSys (CoordSys::*) (const Matrix3d&, const Vector3d&, const CoordSys&) const) &CoordSys::rotateLocal)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const CoordSys& cs) { return py::make_tuple(cs.m_origin, cs.m_rot); },
                [](py::tuple t) {
                    return CoordSys(
                        t[0].cast<Vector3d>(), t[1].cast<Matrix3d>()
                    );
                }
            ))
            .def("__hash__", [](const CoordSys& cs) {
                auto result = py::hash(py::make_tuple("CoordSys"));
                const double* d = &cs.m_origin[0];
                for (int i=0; i<3; i++)
                    result = 1000003*result ^ py::hash(py::float_(d[i]));
                d = &cs.m_rot(0,0);
                for (int i=0; i<9; i++)
                    result = 1000003*result ^ py::hash(py::float_(d[i]));
                result = (result == -1) ? -2 : result;
                return result;
            });
    }


    // Version of applyForward that accepts three congruent numpy arrays (x, y, z), and returns
    // three transformed numpy arrays with the new coordinates.
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
    numpyApplyForward(const CoordTransform& ct,
                      py::array_t<double> xs,
                      py::array_t<double> ys,
                      py::array_t<double> zs) {
        auto bufX = xs.request();
        auto bufY = ys.request();
        auto bufZ = zs.request();
        if (bufX.ndim != bufY.ndim || bufX.ndim != bufZ.ndim)
            throw std::runtime_error("Dimensions must match");
        if (bufX.size != bufY.size || bufX.size != bufZ.size)
            throw std::runtime_error("Sizes much match");
        auto resultX = py::array_t<double>(bufX.shape, bufX.strides);
        auto resultY = py::array_t<double>(bufY.shape, bufY.strides);
        auto resultZ = py::array_t<double>(bufZ.shape, bufZ.strides);
        auto bufXOut = resultX.request();
        auto bufYOut = resultY.request();
        auto bufZOut = resultZ.request();

        double *ptrX = (double *) bufX.ptr;
        double *ptrY = (double *) bufY.ptr;
        double *ptrZ = (double *) bufZ.ptr;
        double *ptrXOut = (double *)  bufXOut.ptr;
        double *ptrYOut = (double *)  bufYOut.ptr;
        double *ptrZOut = (double *)  bufZOut.ptr;

        auto v = Vector3d();
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            v = ct.applyForward(Vector3d(ptrX[idx], ptrY[idx], ptrZ[idx]));
            ptrXOut[idx] = v[0];
            ptrYOut[idx] = v[1];
            ptrZOut[idx] = v[2];
        }
        return std::make_tuple(resultX, resultY, resultZ);
    }

    // Version of applyReverse that accepts three congruent numpy arrays (x, y, z), and returns
    // three transformed numpy arrays with the new coordinates.
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
    numpyApplyReverse(const CoordTransform& ct,
                      py::array_t<double> xs,
                      py::array_t<double> ys,
                      py::array_t<double> zs) {
        auto bufX = xs.request();
        auto bufY = ys.request();
        auto bufZ = zs.request();
        if (bufX.ndim != bufY.ndim || bufX.ndim != bufZ.ndim)
            throw std::runtime_error("Dimensions must match");
        if (bufX.size != bufY.size || bufX.size != bufZ.size)
            throw std::runtime_error("Sizes much match");
        auto resultX = py::array_t<double>(bufX.shape, bufX.strides);
        auto resultY = py::array_t<double>(bufY.shape, bufY.strides);
        auto resultZ = py::array_t<double>(bufZ.shape, bufZ.strides);
        auto bufXOut = resultX.request();
        auto bufYOut = resultY.request();
        auto bufZOut = resultZ.request();

        double *ptrX = (double *) bufX.ptr;
        double *ptrY = (double *) bufY.ptr;
        double *ptrZ = (double *) bufZ.ptr;
        double *ptrXOut = (double *)  bufXOut.ptr;
        double *ptrYOut = (double *)  bufYOut.ptr;
        double *ptrZOut = (double *)  bufZOut.ptr;

        auto v = Vector3d();
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            v = ct.applyReverse(Vector3d(ptrX[idx], ptrY[idx], ptrZ[idx]));
            ptrXOut[idx] = v[0];
            ptrYOut[idx] = v[1];
            ptrZOut[idx] = v[2];
        }
        return std::make_tuple(resultX, resultY, resultZ);
    }

    void pyExportCoordTransform(py::module& m) {
        py::class_<CoordTransform, std::shared_ptr<CoordTransform>>(m, "CoordTransform")
            .def(py::init<const CoordSys&, const CoordSys&>())
            .def("applyForward", (Vector3d (CoordTransform::*)(const Vector3d&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Vector3d (CoordTransform::*)(const Vector3d&) const) &CoordTransform::applyReverse)
            .def("applyForward", [](const CoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyForward(ct, xs, ys, zs);
            })
            .def("applyReverse", [](const CoordTransform& ct, py::array_t<double> xs, py::array_t<double> ys, py::array_t<double> zs){
                return numpyApplyReverse(ct, xs, ys, zs);
            })
            .def("applyForward", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyForward)
            .def("applyReverse", (Ray (CoordTransform::*)(const Ray&) const) &CoordTransform::applyReverse)
            .def("applyForward", [](const CoordTransform& ct, const RayVector& rv){
                RayVector result;
                result.rays = std::move(ct.applyForward(rv.rays));
                return result;
            })
            .def("applyReverse", [](const CoordTransform& ct, const RayVector& rv){
                RayVector result;
                result.rays = std::move(ct.applyReverse(rv.rays));
                return result;
            })
            .def("applyForwardInPlace", (void (CoordTransform::*)(Ray&) const) &CoordTransform::applyForwardInPlace)
            .def("applyForwardInPlace", [](const CoordTransform& ct, RayVector& rv){
                ct.applyForwardInPlace(rv.rays);
            })
            .def("applyReverseInPlace", (void (CoordTransform::*)(Ray&) const) &CoordTransform::applyReverseInPlace)
            .def("applyReverseInPlace", [](const CoordTransform& ct, RayVector& rv){
                ct.applyReverseInPlace(rv.rays);
            })
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const CoordTransform& ct) { return py::make_tuple(ct.getDr(), ct.getRot()); },
                [](py::tuple t) { return CoordTransform(t[0].cast<Vector3d>(), t[1].cast<Matrix3d>()); }
            ))
            .def("__hash__", [](CoordTransform& ct) {
                auto result = py::hash(py::make_tuple("CoordTransform"));
                const double* d = &ct.getDr()[0];
                for (int i=0; i<3; i++)
                    result = 1000003*result ^ py::hash(py::float_(d[i]));
                d = &ct.getRot()(0,0);
                for (int i=0; i<9; i++)
                    result = 1000003*result ^ py::hash(py::float_(d[i]));
                result = (result == -1) ? -2 : result;
                return result;
            })
            .def("__repr__", &CoordTransform::repr);
    }
}
