#ifndef batoid_utils_h
#define batoid_utils_h

#include <iterator>
#include <future>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace batoid {

    /// solve a x^2 + b x + c == 0 for x robustly.
    /// return value is number of solutions.
    /// if only 1 solution is found, result is placed in r1
    /// if 2 solutions are found, results are placed in r1 and r2 with r1 <= r2.
    int solveQuadratic(double a, double b, double c, double& r1, double& r2);

    // InputIt models a RandomAccessIterator
    // OutputIt models a RandomAccessIterator
    // Note that the output must already be allocated!  I.e., no back_inserter
    // should be used for d_first.
    template<typename InputIt, typename OutputIt, typename UnaryOperation>
    void chunkedParallelTransform(
        InputIt first1, InputIt last1, OutputIt d_first,
        UnaryOperation unary_op,
        typename std::iterator_traits<InputIt>::difference_type chunksize)
    {
        auto len = last1 - first1;
        if (len <= chunksize) {
            std::transform(first1, last1, d_first, unary_op);
        } else {
            InputIt mid = first1 + len/2;
            OutputIt d_mid = d_first + len/2;
            auto handle = std::async(std::launch::async,
                                     chunkedParallelTransform<InputIt, OutputIt, UnaryOperation>,
                                     mid, last1, d_mid, unary_op, chunksize);
            chunkedParallelTransform(first1, mid, d_first, unary_op, chunksize);
            handle.wait();
        }
    }

    // Same as above, but for a binary operation
    template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOperation>
    void chunkedParallelTransform(
        InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first,
        BinaryOperation binary_op,
        typename std::iterator_traits<InputIt1>::difference_type chunksize)
    {
        auto len = last1 - first1;
        if (len <= chunksize) {
            std::transform(first1, last1, first2, d_first, binary_op);
        } else {
            InputIt1 mid1 = first1 + len/2;
            InputIt2 mid2 = first2 + len/2;
            OutputIt d_mid = d_first + len/2;

            auto handle = std::async(std::launch::async,
                                     chunkedParallelTransform<InputIt1, InputIt2, OutputIt, BinaryOperation>,
                                     mid1, last1, mid2, d_mid, binary_op, chunksize);
            chunkedParallelTransform(first1, mid1, first2, d_first, binary_op, chunksize);
            handle.wait();
        }
    }

    // Now parallelize for_each in addition to transform
    template<typename It, typename UnaryOperation>
    void chunked_parallel_for_each(
        It first, It last, UnaryOperation unary_op,
        typename std::iterator_traits<It>::difference_type chunksize)
    {
        auto len = last - first;
        if (len <= chunksize) {
            std::for_each(first, last, unary_op);
        } else {
            It mid = first + len/2;
            auto handle = std::async(std::launch::async,
                                     chunked_parallel_for_each<It, UnaryOperation>,
                                     mid, last, unary_op, chunksize);
            chunked_parallel_for_each(first, mid, unary_op, chunksize);
            handle.wait();
        }
    }

    // A binary operator version of for_each
    template<typename It1, typename It2, typename BinaryOperation>
    BinaryOperation for_each(
        It1 first1, It1 last1, It2 first2, BinaryOperation binary_op)
    {
        for(; first1 != last1; ++first1, ++first2) {
            binary_op(*first1, *first2);
        }
        return binary_op;
    }

    template<typename It1, typename It2, typename BinaryOperation>
    void chunked_parallel_for_each(
        It1 first1, It1 last1, It2 first2, BinaryOperation binary_op,
        typename std::iterator_traits<It1>::difference_type chunksize)
    {
        auto len = last1 - first1;
        if (len <= chunksize) {
            for_each(first1, last1, first2, binary_op);
        } else {
            It1 mid1 = first1 + len/2;
            It2 mid2 = first2 + len/2;
            auto handle = std::async(std::launch::async,
                                     chunked_parallel_for_each<It1, It2, BinaryOperation>,
                                     mid1, last1, mid2, binary_op, chunksize);
            chunked_parallel_for_each(first1, mid1, first2, binary_op, chunksize);
            handle.wait();
        }
    }

    // And now some versions that automatically choose a chunksize based on
    // hardware concurrency
    // These currently only really work as intended if nthread is a
    // power of 2, but that seems to be okay for now.

    extern unsigned int nthread;
    void setNThread(unsigned int);
    unsigned int getNThread();

    template<typename InputIt, typename OutputIt, typename UnaryOperation>
    void parallelTransform(
        InputIt first1, InputIt last1, OutputIt d_first,
        UnaryOperation unary_op)
    {
        auto len = last1 - first1;
        len /= nthread;
        // a bit of slop.  We want to be efficient, but not launch more threads than necessary.
        len += 1;
        chunkedParallelTransform(first1, last1, d_first, unary_op, len);
    }

    template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOperation>
    void parallelTransform(
        InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first,
        BinaryOperation binary_op)
    {
        auto len = last1 - first1;
        len /= nthread;
        // a bit of slop.  We want to be efficient, but not launch more threads than necessary.
        len += 1;
        chunkedParallelTransform(first1, last1, first2, d_first, binary_op, len);
    }

    template<typename It, typename UnaryOperation>
    void parallel_for_each(
        It first, It last, UnaryOperation unary_op)
    {
        auto len = last - first;
        len /= nthread;
        // a bit of slop.  We want to be efficient, but not launch more threads than necessary.
        len += 1;
        chunked_parallel_for_each(first, last, unary_op, len);
    }

    template<typename It1, typename It2, typename BinaryOperation>
    void parallel_for_each(
        It1 first1, It1 last1, It2 first2, BinaryOperation binary_op)
    {
        auto len = last1 - first1;
        len /= nthread;
        // a bit of slop.  We want to be efficient, but not launch more threads than necessary.
        len += 1;
        chunked_parallel_for_each(first1, last1, first2, binary_op, len);
    }

}

#endif
