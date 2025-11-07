#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

using namespace std;

namespace utils{
    inline double degrees_to_radians(double degrees) { return degrees * M_PI / 180.0; }
    const double inf = numeric_limits<double>::infinity();
    const double PI = 3.1415926535897932385;

    // ANSI color codes for terminal output
    namespace colors {
        const char* const RESET = "\033[0m";
        const char* const RED = "\033[31m";
        const char* const GREEN = "\033[32m";
        const char* const YELLOW = "\033[33m";
        const char* const BLUE = "\033[34m";
        const char* const MAGENTA = "\033[35m";
        const char* const CYAN = "\033[36m";
        const char* const WHITE = "\033[37m";
        const char* const BOLD_RED = "\033[1;31m";
    }

    // Custom streambuf that adds color prefix
    class ColoredStreamBuf : public std::streambuf {
    private:
        std::streambuf* original_buf;
        const char* color_code;
        bool at_line_start;

    public:
        ColoredStreamBuf(std::streambuf* buf, const char* color) 
            : original_buf(buf), color_code(color), at_line_start(true) {}

        ~ColoredStreamBuf() override = default;

    protected:
        int overflow(int c) override {
            if (at_line_start && c != EOF) {
                // Write color code at start of line
                for (const char* p = color_code; *p; ++p) {
                    original_buf->sputc(*p);
                }
                at_line_start = false;
            }
            
            if (c == '\n') {
                // Write reset code before newline
                for (const char* p = colors::RESET; *p; ++p) {
                    original_buf->sputc(*p);
                }
                at_line_start = true;
            }
            
            return original_buf->sputc(c);
        }

        int sync() override {
            return original_buf->pubsync();
        }
    };

    // Global streambuf instance for cerr coloring
    inline ColoredStreamBuf* colored_cerr_buf = nullptr;

    // Function to enable colored cerr output
    inline void enable_colored_cerr() {
        if (!colored_cerr_buf) {
            colored_cerr_buf = new ColoredStreamBuf(std::cerr.rdbuf(), colors::BOLD_RED);
            std::cerr.rdbuf(colored_cerr_buf);
        }
    }

    // Function to disable colored cerr output (cleanup)
    inline void disable_colored_cerr() {
        if (colored_cerr_buf) {
            delete colored_cerr_buf;
            colored_cerr_buf = nullptr;
        }
    }
};
