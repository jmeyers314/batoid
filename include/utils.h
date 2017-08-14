#ifndef batoid_utils_h
#define batoid_utils_h

#include <iterator>
#include <future>
#include <algorithm>

namespace batoid {

    int solveQuadratic(double a, double b, double c, double& r1, double& r2);

    // InputIt models a RandomAccessIterator
    // OutputIt models a RandomAccessIterator
    template<typename InputIt, typename OutputIt, typename UnaryOperation>
    void parallelTransform(
        InputIt first1, InputIt last1, OutputIt d_first,
        UnaryOperation unary_op,
        typename std::iterator_traits<InputIt>::difference_type chunksize)
    {
        auto len = last1 - first1;
        if (len < chunksize) {
            std::transform(first1, last1, d_first, unary_op);
        } else {
            InputIt mid = first1 + len/2;
            OutputIt d_mid = d_first + len/2;
            auto handle = std::async(std::launch::async,
                                     parallelTransform<InputIt, OutputIt, UnaryOperation>,
                                     mid, last1, d_mid, unary_op, chunksize);
            parallelTransform(first1, mid, d_first, unary_op, chunksize);
            handle.wait();
        }
    }

    // Same as above, but for a binary operation
    template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOperation>
    void parallelTransform(
        InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first,
        BinaryOperation binary_op,
        typename std::iterator_traits<InputIt1>::difference_type chunksize)
    {
        auto len = last1 - first1;
        if (len < chunksize) {
            std::transform(first1, last1, first2, d_first, binary_op);
        } else {
            InputIt1 mid1 = first1 + len/2;
            InputIt2 mid2 = first2 + len/2;
            OutputIt d_mid = d_first + len/2;

            auto handle = std::async(std::launch::async,
                                     parallelTransform<InputIt1, InputIt2, OutputIt, BinaryOperation>,
                                     mid1, last1, mid2, d_mid, binary_op, chunksize);
            parallelTransform(first1, mid1, first2, d_first, binary_op, chunksize);
            handle.wait();
        }
    }
}

#endif
