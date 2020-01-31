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
        EqArgVec(const std::vector<double>& args, const double slopFrac) :
            vec(args), da(vec[1]-vec[0]), slop(da*slopFrac) {}

        int upperIndex(const double a) const;

        // pass through a few std::vector methods.
        citer begin() {return vec.cbegin();}
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
        Bicubic(
            const std::vector<double> xs,
            const std::vector<double> ys,
            const DRef<MatrixXd> zs,
            const DRef<MatrixXd> dzdxs,
            const DRef<MatrixXd> dzdys,
            const DRef<MatrixXd> d2zdxdys,
            const double slopFrac
        );
        virtual double sag(double, double) const override;
        virtual Vector3d normal(double, double) const override;
        virtual bool timeToIntersect(const Ray& r, double& t) const override;

    private:
        double oneDSpline(double x, double val0, double val1, double der0, double der1) const;
        double oneDGrad(double x, double val0, double val1, double der0, double der1) const;

        const EqArgVec _xargs;
        const EqArgVec _yargs;
        const DRef<MatrixXd> _zs;
        const DRef<MatrixXd> _dzdxs;
        const DRef<MatrixXd> _dzdys;
        const DRef<MatrixXd> _d2zdxdys;
    };

}
#endif
