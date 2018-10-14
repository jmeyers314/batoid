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
            InputIt mid = first1 + chunksize;
            OutputIt d_mid = d_first + chunksize;
            auto handle = std::async(std::launch::async,
                                     chunkedParallelTransform<InputIt, OutputIt, UnaryOperation>,
                                     mid, last1, d_mid, unary_op, chunksize);
            std::transform(first1, mid, d_first, unary_op);
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
            InputIt1 mid1 = first1 + chunksize;
            InputIt2 mid2 = first2 + chunksize;
            OutputIt d_mid = d_first + chunksize;
            auto handle = std::async(std::launch::async,
                                     chunkedParallelTransform<InputIt1, InputIt2, OutputIt, BinaryOperation>,
                                     mid1, last1, mid2, d_mid, binary_op, chunksize);
            transform(first1, mid1, first2, d_first, binary_op);
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
            It mid = first + chunksize;
            auto handle = std::async(std::launch::async,
                                     chunked_parallel_for_each<It, UnaryOperation>,
                                     mid, last, unary_op, chunksize);
            std::for_each(first, mid, unary_op);
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
            It1 mid1 = first1 + chunksize;
            It2 mid2 = first2 + chunksize;
            auto handle = std::async(std::launch::async,
                                     chunked_parallel_for_each<It1, It2, BinaryOperation>,
                                     mid1, last1, mid2, binary_op, chunksize);
            std::for_each(first1, mid1, first2, binary_op);
            handle.wait();
        }
    }

    // And now some versions that automatically choose a chunksize based on
    // hardware concurrency

    extern unsigned int nthread;
    extern unsigned int minChunk;

    void setNThread(unsigned int);
    unsigned int getNThread();

    void setMinChunk(unsigned int);
    unsigned int getMinChunk();

    template<typename InputIt, typename OutputIt, typename UnaryOperation>
    void parallelTransform(
        InputIt first1, InputIt last1, OutputIt d_first,
        UnaryOperation unary_op)
    {
        unsigned int len = last1 - first1;
        len /= nthread;
        len = std::max(len+1, minChunk);
        chunkedParallelTransform(first1, last1, d_first, unary_op, len);
    }

    template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOperation>
    void parallelTransform(
        InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first,
        BinaryOperation binary_op)
    {
        unsigned int len = last1 - first1;
        len /= nthread;
        len = std::max(len+1, minChunk);
        chunkedParallelTransform(first1, last1, first2, d_first, binary_op, len);
    }

    template<typename It, typename UnaryOperation>
    void parallel_for_each(
        It first, It last, UnaryOperation unary_op)
    {
        unsigned int len = last - first;
        len /= nthread;
        len = std::max(len+1, minChunk);
        chunked_parallel_for_each(first, last, unary_op, len);
    }

    template<typename It1, typename It2, typename BinaryOperation>
    void parallel_for_each(
        It1 first1, It1 last1, It2 first2, BinaryOperation binary_op)
    {
        unsigned int len = last1 - first1;
        len /= nthread;
        len = std::max(len+1, minChunk);
        chunked_parallel_for_each(first1, last1, first2, binary_op, len);
    }

}

#endif
