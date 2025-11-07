/**
 * @class Interval
 * @brief Represents a mathematical interval [min, max] with utility functions.
 *
 * The Interval class provides a way to represent and manipulate intervals
 * on the real number line. It includes methods for checking containment,
 * clamping values, and determining the size of the interval.
 *
 * The class also defines two special intervals:
 * - `Interval::empty`: Represents an empty interval with min = +inf and max = -inf.
 * - `Interval::universe`: Represents the entire real number line with min = -inf and max = +inf.
 */

#pragma once

#include "utils.h"

using namespace utils;

class Interval
{
 public:
   double min, max;

   Interval() : min(+inf), max(-inf) {} // Default interval is empty

   Interval(double min, double max) : min(min), max(max) {}

   double size() const { return max - min; }

   bool contains(double x) const { return min <= x && x <= max; }

   bool surrounds(double x) const { return min < x && x < max; }

   double clamp(double x) const { return std::max(min, std::min(x, max)); }

   static const Interval empty, universe;
};

const Interval Interval::empty = Interval(+inf, -inf);
const Interval Interval::universe = Interval(-inf, +inf);