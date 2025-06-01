
/**
 * @file
 * @brief Execution time measurement
 * @ingroup utilities
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 *

 */

#ifndef TIMER_H
#define TIMER_H

#include <map>
#include <string>
#include <chrono>

using std::exception;

using std::map;
using std::nano;
using std::string;
using std::chrono::duration;
using std::chrono::steady_clock;

/** @brief Timing executiong time. */
class Timer
{
public:
    /**
     * @brief Creates a time point associated with a tag.
     *
     * @param  tag Tag.
     *
     * @returns steady_clock::time_point associated to the tag.
     */

    steady_clock::time_point mark(string tag)
    {
        auto point = now();
        points[tag] = point;
        return point;
    }

    /**
     * @brief Elapsed time from tag.
     *
     * @param  tag Tag.
     *
     * @returns Duration(now - time point of tag)
     */

    duration<float, std::nano> elapsed(string tag)
    {
        duration<float, std::nano> ret{};
        try
        {
            return duration<float, std::nano>(now() - points.at(tag));
        }
        catch (exception &ex)
        {
        }
        return ret;
    }

    /**
     * @brief Elapsed seconds from a tag.
     *
     * @param  tag Tag.
     *
     * @returns Seconds(now - time point of tag).
     */

    float seconds(string tag)
    {
        try
        {
            return duration<float>(elapsed(tag)).count();
        }
        catch (exception &ex)
        {
        }
        return 0;
    }

    /**
     * @brief Elapsed milliseconds from a tag.
     *
     * @param  tag Tag.
     *
     * @returns Milliseconds (now - time point of tag)
     */

    float milliseconds(string tag)
    {
        try
        {
            return duration<float, std::milli>(elapsed(tag)).count();
        }
        catch (exception &ex)
        {
        }
        return 0;
    }

    /**
     * @brief Elapsed microseconds from a tag.
     *
     * @param  tag Tag.
     *
     * @returns Microseconds (now - time point of tag)
     */

    float microseconds(string tag)
    {
        try
        {
            return duration<float, std::micro>(elapsed(tag)).count();
        }
        catch (exception &ex)
        {
        }
        return 0;
    }

    /**
     * @brief Elapsed nanoseconds from a tag.
     *
     * @param  tag Tag.
     *
     * @returns Nanoseconds (now - time point of tag)
     */

    float nanoseconds(string tag)
    {
        try
        {
            return elapsed(tag).count();
        }
        catch (exception &ex)
        {
        }
        return 0;
    }

    /**
     * @brief Gets the current time point - now.
     *
     * @returns steady_clock::time_point.
     */

    static inline steady_clock::time_point now()
    {
        return steady_clock::now();
    }

private:
    map<std::string, steady_clock::time_point> points; /*!< Time points associated with a tag.*/
};

#endif
