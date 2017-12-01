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


#ifndef batoid_table_h
#define batoid_table_h

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <functional>

namespace batoid {

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /// @brief Basic exception class thrown by Table
    class TableError : public std::runtime_error
    {
    public:
        TableError(const std::string& m) : std::runtime_error("Table Error: " +m) {}
    };

    /// @brief Exception class for Table access ouside the allowed range
    class TableOutOfRange : public TableError
    {
        template <typename A>
        static std::string BuildErrorMessage(A val, A min, A max)
        {
            // See discussion in Std.h about this initial value.
            std::ostringstream oss(" ");
            oss << "Argument "<<val<<" out of range ("<<min<<".."<<max<<")";
            return oss.str();
        }
    public:
        template <typename A>
        TableOutOfRange(A val, A min, A max) :
            TableError(BuildErrorMessage(val,min,max)) {}
    };

    /// @brief Exception class for I/O errors when reading in a Table
    class TableReadError : public TableError
    {
    public:
        TableReadError(const std::string& c) : TableError("Data read error for line ->"+c) {}
    };

    //! @endcond

    /**
     * @brief A class to represent an argument vector for a Table or Table2D.
     *
     * Basically a std::vector with a few extra bells and whistles to deal with potentially
     * equally-spaced arguments, upper and lower slop, and fast indexing.
     */
    template<class A>
    class ArgVec
    {
    public:
        ArgVec() : isReady(false) {}
        ArgVec(const std::vector<A>& args) : vec(args), isReady(false) {}
        template<class InputIterator>
        ArgVec(InputIterator first, InputIterator last) : vec(first, last), isReady(false) {}

        int upperIndex(const A a) const;

        // pass through a few std::vector methods.
        typename std::vector<A>::iterator begin() {return vec.begin();}
        typename std::vector<A>::iterator end() {return vec.end();}
        const A& front() const {return vec.front();}
        const A& back() const {return vec.back();}
        const A& operator[](int i) const {return vec[i];}
        typename std::vector<A>::iterator insert(typename std::vector<A>::iterator it, const A a);
        size_t size() const {return vec.size();}

        const std::vector<A>& getArgs() const { return vec; }

    private:
        typedef typename std::vector<A>::const_iterator citer;
        std::vector<A> vec;
        mutable bool isReady;
        // A few convenient additional member variables.
        mutable A lower_slop, upper_slop;
        mutable bool equalSpaced;
        mutable A da;
        mutable int lastIndex;
        void setup() const;
    };

    /**
     * @brief A class to represent lookup tables for a function y = f(x).
     *
     * A is the type of the argument of the function.
     * V is the type of the value of the function.
     *
     * Requirements for A,V:
     *   A must have ordering operators (< > ==) and the normal arithmetic ops (+ - * /)
     *   V must have + and *.
     */
    template<class V, class A>
    class Table
    {
    public:
        enum class interpolant { linear, floor, ceil, nearest };

        /// Table from args, vals
        Table(const A* _args, const V* _vals, int N, interpolant in) :
                iType(in), args(_args, _args+N), vals(_vals, _vals+N), isReady(false) {}
        Table(const std::vector<A>& _args, const std::vector<V>& _vals, interpolant in) :
                iType(in), args(_args), vals(_vals), isReady(false) {}
        /// Empty Table
        Table(interpolant in) : iType(in), isReady(false) {}

        void init(); /// Common initialization code.

        A argMin() const {return args.front();}
        A argMax() const {return args.back();}

        /// Return the size of the table.
        int size() const {return vals.size();}

        /// Insert an (x, y(x)) pair into the table.
        void addEntry(const A arg, const V val);

        /// interp, return V(0) if beyond bounds
        V operator()(const A a) const;

        /// interp, but exception if beyond bounds
        V lookup(const A a) const;

        /// interp many values at once
        void interpMany(const A* argvec, V* valvec, int N) const;

        const std::vector<A>& getArgs() const { return args.getArgs(); }
        const std::vector<V>& getVals() const { return vals; }
        int getN() const {return vals.size();}
        interpolant getInterp() const { return iType; }

    private:
        interpolant iType;
        ArgVec<A> args;

        std::vector<V> vals;
        mutable std::vector<V> y2;
        mutable bool isReady;

        typedef V (Table<V,A>::*TableMemFn)(const A x, int i) const;
        mutable TableMemFn interpolate;
        V linearInterpolate(const A a, int i) const;
        V floorInterpolate(const A a, int i) const;
        V ceilInterpolate(const A a, int i) const;
        V nearestInterpolate(const A a, int i) const;

        void setup() const;
    };

    template<class V, class A>
    bool operator==(const Table<V,A>& t1, const Table<V,A>& t2);
    template<class V, class A>
    bool operator!=(const Table<V,A>& t1, const Table<V,A>& t2);
}

#endif
