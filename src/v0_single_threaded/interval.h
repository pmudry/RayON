#pragma once

#include "utils.h"

class interval {
  public:
    double min, max;

    interval() : min(+inf), max(-inf) {} // Default interval is empty

    interval(double min, double max) : min(min), max(max) {}

    double size() const {
        return max - min;
    }

    bool contains(double x) const {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const {
        return min < x && x < max;
    }

    static const interval empty, universe;
};

const interval interval::empty    = interval(+inf, -inf);
const interval interval::universe = interval(-inf, +inf);