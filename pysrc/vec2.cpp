#include <stdexcept>
#include "vec2.h"
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportVec2(py::module& m) {
        PYBIND11_NUMPY_DTYPE(Vec2, x, y);
        py::class_<Vec2>(m, "Vec2", "")
            .def(py::init<double,double>(), "init", "x"_a, "y"_a)
            .def(py::init<>())
            .def(py::init<std::array<double,2>>())
            .def("MagnitudeSquared", &Vec2::MagnitudeSquared, "Return square of vector magnitude.")
            .def("Magnitude", &Vec2::Magnitude, "Return vector magnitude.")
            .def("UnitVec2", &Vec2::UnitVec2, "Return unit vector pointing in same direction.")
            .def("__repr__", &Vec2::repr)
            .def_readonly("x", &Vec2::x, "x-coordinate of vector")
            .def_readonly("y", &Vec2::y, "y-coordinate of vector")
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(py::self - py::self)
            .def(py::self -= py::self)
            .def(py::self * double())
            .def(py::self *= double())
            .def(py::self / double())
            .def(py::self /= double())
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(-py::self)
            .def(py::pickle(
                [](const Vec2& v){ // __getstate__
                    return py::make_tuple(v.x, v.y);
                },
                [](py::tuple t){ // __setstate__
                    if (t.size() != 2)
                        throw std::runtime_error("Invalid state!");
                    return Vec2(t[0].cast<double>(), t[1].cast<double>());
                }
            ))
            .def("__hash__", [](const Vec2& r) { return py::hash(py::make_tuple("Ray", r.x, r.y)); });

        py::class_<Rot2>(m, "Rot2", py::buffer_protocol())
            .def_buffer([](Rot2& r) -> py::buffer_info {
                return py::buffer_info(
                    r.data.data(),
                    sizeof(double),
                    py::format_descriptor<double>::format(),
                    2,
                    {2, 2},
                    {sizeof(double) * 2, sizeof(double)}
                );
            })
            .def(py::init<>())
            .def(py::init<std::array<double,4>>())
            .def("__repr__", &Rot2::repr)
            .def("determinant", &Rot2::determinant)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const Rot2& r) { // __getstate__
                    return py::make_tuple(r.data);
                },
                [](py::tuple t){ // __setstate__
                    if (t.size() != 1)
                        throw std::runtime_error("Invalid state!");
                    return Rot2(t[0].cast<std::array<double,4>>());
                }
            ))
            // See http://effbot.org/zone/python-hash.htm#tuples
            .def("__hash__", [](const Rot2& r) {
                auto result = py::hash(py::str("Rot2"));
                for (const auto& elt : r.data)
                    result = 1000003*result ^ py::hash(py::float_(elt));
                result ^= r.data.size();
                result = (result == -1) ? -2 : result;
                return result;
            });

        m.def("DotProduct", &DotProduct);

        m.def("RotVec", &RotVec);
        m.def("UnRotVec", &UnRotVec);
    }
}
