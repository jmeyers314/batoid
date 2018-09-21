#ifndef batoid_bicubic_h
#define batoid_bicubic_h

#include <vector>
#include <sstream>
#include <limits>
#include "surface.h"
#include "ray.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Vector3d;
template<class T>
using DRef = Eigen::Ref<T, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

namespace batoid {

    // ArgVec class known to be equal-spaced.
    class EqArgVec
    {
    private:
        typedef typename std::vector<double>::const_iterator citer;
    public:
        EqArgVec(const std::vector<double>& args) :
            vec(args), da(vec[1]-vec[0]), slop(da*1e-6) {}

        int upperIndex(const double a) const;

        // pass through a few std::vector methods.
        // typename std::vector<double>::iterator begin() {return vec.cbegin();}
        citer begin() {return vec.cbegin();}
        // typename std::vector<double>::iterator end() {return vec.cend();}
        citer end() {return vec.cend();}
        const double& front() const {return vec.front();}
        const double& back() const {return vec.back();}
        const double& operator[](int i) const {return vec[i];}
        typename std::vector<double>::iterator insert(
            typename std::vector<double>::iterator it, const double a);
        size_t size() const {return vec.size();}

        const std::vector<double>& getArgs() const { return vec; }

        const double getDa() const { return da; }
    private:
        const std::vector<double> vec;
        const double da;
        const double slop;
    };

    class Bicubic : public Surface {
    public:
        // Bicubic(std::vector<double> xs, std::vector<double> ys, MatrixXd zs);
        Bicubic(std::vector<double> xs, std::vector<double> ys, DRef<MatrixXd> zs);
        virtual double sag(double, double) const override;
        virtual Vector3d normal(double, double) const override;
        bool timeToIntersect(const Ray& r, double& t) const override;

    private:
        const EqArgVec _xargs;
        const EqArgVec _yargs;
        // const MatrixXd _zs;
        const DRef<MatrixXd> _zs;
    };

}
#endif
