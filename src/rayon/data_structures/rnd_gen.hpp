/**
 * @class RndGen
 * @brief A random number generator utility class.
 *
 * This class provides static methods to generate random numbers, including uniform and normal distributions.
 * It uses the Mersenne Twister engine for random number generation and allows setting a seed for reproducibility.
 * The random number generator is thread-local to ensure thread safety in multi-threaded applications.
 */
#include <random>

#pragma once
using namespace std;

class RndGen
{
 public:
   RndGen() = delete; // Prevent instantiation

   static void set_seed(unsigned int seed) { get_rng() = std::mt19937(seed); }

   static double random_double()
   {
      // Returns a random real in [0,1).
      static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
      return distribution(get_rng());
   }

   static double random_double(double min, double max) { return random_double() * (max - min) + min; }

 private:
   static std::mt19937 &get_rng()
   {
      static thread_local std::mt19937 gen(std::random_device{}());
      return gen;
   }
};