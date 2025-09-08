// Random number utility functions
#include <random>
#include <functional>

// default_random_engine generator;

// Global random number generator
inline std::mt19937& get_rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

inline unsigned int get_random_seed() {
    static const unsigned int seed = std::random_device{}();
    return seed;
}

inline double random_double() {
    // Returns a random real in [0,1).
    static uniform_real_distribution<double> distribution(0.0, 1.0);
    static auto gen = bind(distribution, get_rng());
    return gen();
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    static uniform_real_distribution<double> distribution(min, max);
    static auto gen = bind(distribution, get_rng());
    return gen();
}

inline double random_normal() {
    // Returns a random real from standard normal distribution (mean=0, stddev=1).
    static std::normal_distribution<double> dis(0.0, 1.0);
    return dis(get_rng());
}

inline double random_normal(double mean, double stddev) {
    // Returns a random real from normal distribution with given mean and standard deviation.
    std::normal_distribution<double> dis(mean, stddev);
    return dis(get_rng());
}