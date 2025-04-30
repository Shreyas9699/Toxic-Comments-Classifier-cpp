#pragma once
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

class Timer
{
public:
    Timer(const std::string& phaseName)
        : name(phaseName), start(std::chrono::high_resolution_clock::now())
    {}

    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "[" << name << "] took "
            << formatHMS(elapsed)
            << "\n";
    }

private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

    // Breaks any chrono::duration into H, M, S and builds a string like "1h 45m 56s" or "10m 25s"
    template<typename Rep, typename Period>
    static std::string formatHMS(const std::chrono::duration<Rep, Period>& dur)
    {
        using namespace std::chrono;
        // count() returns seconds::rep (typically int64_t)
        auto total_secs = duration_cast<seconds>(dur).count();

        // keep the same type, no narrowing warnings
        auto hours = total_secs / 3600;
        total_secs %= 3600;
        auto minutes = total_secs / 60;
        auto seconds = total_secs % 60;

        std::ostringstream oss;
        if (hours > 0) oss << hours << "h ";
        if (minutes > 0) oss << minutes << "m ";
        oss << seconds << "s";

        return oss.str();
    }
};
