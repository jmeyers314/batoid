// Note, this file mostly stolen from GalSim.  It had the following copyright
// notice attached:

/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#include "table.h"
#include <cmath>
#include <vector>

#include <iostream>

namespace batoid {

    // ArgVec

    template<class A>
    void ArgVec<A>::setup() const
    {
        int N = vec.size();
        const double tolerance = 0.01;
        da = (vec.back() - vec.front()) / (N-1);
        if (da == 0.) throw TableError("First and last arguments are equal.");
        equalSpaced = true;
        for (int i=1; i<N; i++) {
            if (std::abs((vec[i] - vec.front())/da - i) > tolerance) equalSpaced = false;
            if (vec[i] <= vec[i-1])
                throw TableError("Table arguments not strictly increasing.");
        }
        lastIndex = 1;
        lower_slop = (vec[1]-vec[0]) * 1.e-6;
        upper_slop = (vec[N-1]-vec[N-2]) * 1.e-6;
        isReady = true;
    }

    // Look up an index.  Use STL binary search.
    template<class A>
    int ArgVec<A>::upperIndex(const A a) const
    {
        if (!isReady) setup();
        if (a<vec.front()-lower_slop || a>vec.back()+upper_slop)
            throw TableOutOfRange(a,vec.front(),vec.back());
        // check for slop
        if (a < vec.front()) return 1;
        if (a > vec.back()) return vec.size()-1;

        if (equalSpaced) {
            int i = int( std::ceil( (a-vec.front()) / da) );
            if (i >= int(vec.size())) --i; // in case of rounding error
            if (i == 0) ++i;
            // check if we need to move ahead or back one step due to rounding errors
            while (a > vec[i]) ++i;
            while (a < vec[i-1]) --i;
            return i;
        } else {
            // xassert(lastIndex >= 1);
            // xassert(lastIndex < vec.size());

            if ( a < vec[lastIndex-1] ) {
                // xassert(lastIndex-2 >= 0);
                // Check to see if the previous one is it.
                if (a >= vec[lastIndex-2]) return --lastIndex;
                else {
                    // Look for the entry from 0..lastIndex-1:
                    citer p = std::upper_bound(vec.begin(), vec.begin()+lastIndex-1, a);
                    // xassert(p != vec.begin());
                    // xassert(p != vec.begin()+lastIndex-1);
                    lastIndex = p-vec.begin();
                    return lastIndex;
                }
            } else if (a > vec[lastIndex]) {
                // xassert(lastIndex+1 < vec.size());
                // Check to see if the next one is it.
                if (a <= vec[lastIndex+1]) return ++lastIndex;
                else {
                    // Look for the entry from lastIndex..end
                    citer p = std::lower_bound(vec.begin()+lastIndex+1, vec.end(), a);
                    // xassert(p != vec.begin()+lastIndex+1);
                    // xassert(p != vec.end());
                    lastIndex = p-vec.begin();
                    return lastIndex;
                }
            } else {
                // Then lastIndex is correct.
                return lastIndex;
            }
        }
    }

    template<class A>
    typename std::vector<A>::iterator ArgVec<A>::insert(
            typename std::vector<A>::iterator it, const A a)
    {
        isReady = false;
        return vec.insert(it, a);
    }


    // Table

    template<class V, class A>
    void Table<V,A>::addEntry(const A a, const V v)
    {
        typename std::vector<A>::const_iterator p = std::upper_bound(args.begin(), args.end(), a);
        int i = p - args.begin();
        args.insert(args.begin()+i, a);
        vals.insert(vals.begin()+i, v);
        isReady = false;
    }

    template<class V, class A>
    void Table<V,A>::setup() const
    {
        if (isReady) return;

        if (vals.size() != args.size())
            throw TableError("args and vals lengths don't match");
        if (vals.size() < 2)
            throw TableError("input vectors are too short for interpolation");
        switch (iType) {
          case interpolant::linear:
               interpolate = &Table<V,A>::linearInterpolate;
               break;
          case interpolant::floor:
               interpolate = &Table<V,A>::floorInterpolate;
               break;
          case interpolant::ceil:
               interpolate = &Table<V,A>::ceilInterpolate;
               break;
          case interpolant::nearest:
               interpolate = &Table<V,A>::nearestInterpolate;
               break;
          default:
               throw TableError("interpolation method not yet implemented");
        }
        isReady = true;
    }

    //lookup and interpolate function value.
    template<class V, class A>
    V Table<V,A>::operator()(const A a) const
    {
        setup();
        if (a<argMin() || a>argMax()) return V(0);
        else {
            int i = args.upperIndex(a);
            return (this->*interpolate)(a, i);
        }
    }

    //lookup and interpolate function value.
    template<class V, class A>
    V Table<V,A>::lookup(const A a) const
    {
        setup();
        int i = args.upperIndex(a);
        return (this->*interpolate)(a, i);
    }

    //lookup and interpolate an array of function values.
    template<class V, class A>
    void Table<V,A>::interpMany(const A* argvec, V* valvec, int N) const
    {
        setup();
        int i;
        for (int k=0; k<N; k++) {
            i = args.upperIndex(argvec[k]);
            valvec[k] = (this->*interpolate)(argvec[k], i);
        }
    }

    template<class V, class A>
    V Table<V,A>::linearInterpolate(const A a, int i) const
    {
        A ax = (args[i] - a) / (args[i] - args[i-1]);
        A bx = 1.0 - ax;
        return vals[i]*bx + vals[i-1]*ax;
    }

    template<class V, class A>
    V Table<V,A>::floorInterpolate(const A a, int i) const
    {
        // On entry, it is only guaranteed that args[i-1] <= a <= args[i].
        // Normally those ='s are ok, but for floor and ceil we make the extra
        // check to see if we should choose the opposite bound.
        if (a == args[i]) i++;
        return vals[i-1];
    }

    template<class V, class A>
    V Table<V,A>::ceilInterpolate(const A a, int i) const
    {
        if (a == args[i-1]) i--;
        return vals[i];
    }

    template<class V, class A>
    V Table<V,A>::nearestInterpolate(const A a, int i) const
    {
        if ((a - args[i-1]) < (args[i] - a)) i--;
        return vals[i];
    }

    template<class V, class A>
    bool operator==(const Table<V,A>& t1, const Table<V,A>& t2) {
        return t1.getArgs() == t2.getArgs() &&
               t1.getVals() == t2.getVals() &&
               t1.getInterp() == t2.getInterp();
    }

    template<class V, class A>
    bool operator!=(const Table<V,A>& t1, const Table<V,A>& t2) {
        return t1.getArgs() != t2.getArgs() ||
               t1.getVals() != t2.getVals() ||
               t1.getInterp() != t2.getInterp();
    }

    // template instantiation
    template class ArgVec<double>;
    template class Table<double, double>;

    template bool operator==<double,double>(const Table<double,double>&, const Table<double,double>&);
    template bool operator!=<double,double>(const Table<double,double>&, const Table<double,double>&);
}
