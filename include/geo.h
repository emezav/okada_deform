/**
 * @file
 * @brief Geospatial classes and functions - WGS84
 * @author Erwin Meza Vega <emezav@unicauca.edu.co> <emezav@gmail.com>
 * @version 1.0
 * @date 2025-06-06
 * @copyright MIT License
 *
 */
#ifndef GEO_H
#define GEO_H

/** @brief Prevent the inclusion of min and max macros (Windows) */
#define NOMINMAX

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

// OS-specific
#ifdef _MSC_VER
// Windows
#include <windows.h>

#else
// Linux
// enable large file support on 32 bit systems
#ifndef _LARGEFILE64_SOURCE

/// Force large file support
#define _LARGEFILE64_SOURCE
#endif
#ifdef _FILE_OFFSET_BITS
#undef _FILE_OFFSET_BITS
#endif
/// Force file offset to 64 bits
#define _FILE_OFFSET_BITS 64
// and include required headers
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

using std::map;
using std::ofstream;
using std::string;
using std::tuple;
using std::vector;
namespace fs = std::filesystem;

using std::cerr;
using std::cout;
using std::endl;

/**
 * @brief Geospatial classes and functions
 */
namespace geo
{

/** @brief Debug flag */
#ifdef _DEBUG
    static inline bool geoDebug{true};
#else
    static inline bool geoDebug{false};
#endif

    /** @brief Value of pi */
    static constexpr auto pi{3.14159265358979323846};

    /** @brief Earth radius */
    static constexpr auto earthRadius{6378137.0f};

    /** @brief Flattening factor of the Earth  - 1 / f*/
    static constexpr auto earthFlattening{1.0f / 298.257223563f};

    /** @brief String operations. */
    struct Strings
    {

        /**
         * @brief Converts string to lowercase
         *
         * @param [in,out] s A string to process.
         *
         * @returns A reference to a string.
         */

        static string &tolower(string &s)
        {
            // convert string to back to lower case
            std::for_each(s.begin(), s.end(), [](char &c)
                          { c = std::tolower(c); });

            return s;
        }

        /**
         * @brief Converts string to lowercase
         *
         * @param [in,out] s A string to process.
         *
         * @returns A reference to a string.
         */

        static string &toupper(string &s)
        {
            // convert string to back to lower case
            std::for_each(s.begin(), s.end(), [](char &c)
                          { c = std::toupper(c); });

            return s;
        }

        /**
         * @brief trim from start (in place)
         * @param [in,out] s A string to process.
         *
         * @returns A reference to a string.
         */

        static string &ltrim(string &s)
        {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch)
                                            { return !std::isspace(ch); }));
            return s;
        }

        /**
         * @brief trim from start (in place)
         *
         * @param [in,out] s		   A string to process.
         * @param 		   except_func The except function.
         *
         * @returns A reference to a string.
         */

        static string &ltrim(string &s, std::function<bool(int)> except_func)
        {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [except_func](int ch)
                                            { return except_func(ch); }));
            return s;
        }

        /**
         * @brief trim from end (in place)
         *
         * @param [in,out] s A string to process.
         *
         * @returns A reference to a string.
         */

        static string &rtrim(string &s)
        {
            s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch)
                                 { return !std::isspace(ch); })
                        .base(),
                    s.end());

            return s;
        }

        /**
         * @brief trim from end (in place)
         *
         * @param [in,out] s		   A string to process.
         * @param 		   except_func The except function.
         *
         * @returns A reference to a string.
         */

        static string &rtrim(string &s, std::function<bool(int)> except_func)
        {
            s.erase(std::find_if(s.rbegin(), s.rend(), [except_func](int ch)
                                 { return except_func(ch); })
                        .base(),
                    s.end());
            return s;
        }

        /**
         * @brief trim from both ends (in place)
         *
         * @param [in,out] s A string to process.
         *
         * @returns A reference to a string.
         */

        static string &trim(string &s)
        {
            ltrim(s);
            rtrim(s);

            return s;
        }

        /**
         * @brief trim from both ends (in place)
         *
         * @param [in,out] s		   A string to process.
         * @param 		   except_func The except function.
         *
         * @returns A reference to a string.
         */

        static string &trim(string &s, std::function<bool(int)> except_func)
        {
            ltrim(s, except_func);
            rtrim(s, except_func);

            return s;
        }

        /**
         * @brief Splits the given string using std::istringstream delimiters
         *
         * @param  str The string.
         *
         * @returns A vector&lt;string&gt;
         */

        static vector<string> split(const string &str)
        {
            vector<string> ret;

            std::istringstream iss(str);

            string token;
            while (iss >> token)
            {
                ret.push_back(token);
            }
            return ret;
        }

        /**
         * @brief Splits
         *
         * @param  str   The string.
         * @param  delim The delimiter.
         *
         * @returns A vector&lt;string&gt;
         */

        static vector<string> split(const string &str, char delim)
        {
            vector<string> ret;

            std::istringstream iss(str);

            string token;
            while (std::getline(iss, token, delim))
            {
                if (token.length() > 0)
                {
                    ret.push_back(token);
                }
            }
            return ret;
        }

        /**
         * @brief Splits using more than one delimiter
         *
         * @param  str   The string.
         * @param  delim Delimiters as string. Order is irrelevant.
         *
         * @returns A vector&lt;string&gt;
         */

        static vector<string> split(const string &str, string delim)
        {
            vector<string> ret;

            int i = 0;

            while (i < str.length())
            {
                // skip delimiters
                while (i < str.length() && delim.find(str[i]) != string::npos)
                {
                    i++;
                }
                if (i == str.length())
                {
                    // end of string
                    break;
                }

                // Store non-delimiters on a temp string
                string s;
                int j = i;
                while (j < str.length() && delim.find(str[j]) == string::npos)
                {
                    s += str[j];
                    j++;
                }

                // If non empty, store into result vector
                if (s.length() > 0)
                {
                    ret.push_back(s);
                }

                // Update search position
                i = j;
            }
            return ret;
        }

        /**
         * @brief Joins a vector of strings
         *
         * @param strings Vector of strings
         * @param glue Glue
         * @return string
         */
        static string join(const vector<string> &strings, string glue)
        {
            std::ostringstream oss;
            size_t n = strings.size();

            for (size_t i = 0; i < n; i++)
            {
                oss << strings[i];
                if (i < n - 1)
                {
                    oss << glue;
                }
            }
            return oss.str();
        }

        /**
         * @brief Unquotes the given string
         *
         * @param [in,out] str The string.
         *
         * @returns A reference to a string.
         */

        static string &unquote(string &str)
        {
            trim(str, [](int c)
                 { return c != '"'; });
            trim(str, [](int c)
                 { return c != '\''; });
            return str;
        }

        /**
         * @brief Replaces
         *
         * @param  str The string.
         * @param  f   A string to process.
         * @param  r   A string to process.
         *
         * @returns A string.
         */

        static string replace(string str, string f, string r)
        {
            size_t pos = 0;
            if ((pos = str.find(f, pos)) != string::npos)
            {
                str.replace(pos, f.length(), r);
            }

            return str;
        }

        /**
         * @brief Replace all
         *
         * @param  str The string.
         * @param  f   A string to process.
         * @param  r   A string to process.
         *
         * @returns A string.
         */

        static string replaceAll(string str, string f, string r)
        {

            size_t pos = 0;
            while ((pos = str.find(f, pos)) != string::npos)
            {
                str.replace(pos, f.length(), r);
            }
            return str;
        }

        /**
         * @brief Replace last
         *
         * @param  str The string.
         * @param  f   A string to process.
         * @param  r   A string to process.
         *
         * @returns A string.
         */

        static string replaceLast(string str, string f, string r)
        {

            size_t ant = string::npos;
            size_t pos = 0;
            while ((pos = str.find(f, pos)) != string::npos)
            {
                ant = pos;
                pos++;
            }

            if (ant != string::npos)
            {
                str.replace(ant, f.length(), r);
            }

            return str;
        }

        /**
         * @brief Scans the file contents looking for new line characters.
         * @return vector of tuples {offset, length} of lines from the start of data.
         */
        static vector<tuple<const size_t, const size_t>> scan(const string &str)
        {

            vector<tuple<const size_t, const size_t>> lineRanges;

            bool insideLine = false;
            bool isNewLine;

            size_t start = 0;
            size_t end{};
            size_t len = str.length();

            for (size_t p = start; p < len; p++)
            {
                const char c = str[p];

                isNewLine = false;
                // Check if current char is a newline
                if (c == '\r' || c == '\n')
                {
                    isNewLine = true;
                }

                // If current char is newline, check for line before this char
                if (isNewLine)
                {
                    if (insideLine && end >= start)
                    {
                        lineRanges.push_back({start, end - start + 1});
                        insideLine = false;
                    }
                }
                else
                {
                    // If insideToken is false, start a new line
                    if (!insideLine)
                    {
                        // Mark start of line
                        start = p;
                    }
                    // Always mark the end.
                    end = p;
                    // Mark we're inside a token
                    insideLine = true;
                }
            }

            // Check for remaining data at the end
            if (insideLine)
            {
                lineRanges.push_back({start, end - start + 1});
                insideLine = false;
            }
            return lineRanges;
        }

        /**
         * @brief Loads a whole file into a string.
         *
         * @param path File path
         * @return string Contents of the file
         */
        static string loadFile(const string &path)
        {
            string s;

            if (!fs::exists(path) || !fs::is_regular_file(path))
            {
                return s;
            }

            // Get file size
            size_t fileSize = fs::file_size(path);

            // Open file for reading
            FILE *fp = std::fopen(path.c_str(), "r");
            if (fp == nullptr)
            {
                return s;
            }

            const auto read = loadFile(fp, fileSize, s);

            std::fclose(fp);

            if (!read)
            {
                s.clear();
                return "";
            }

            if (read < fileSize)
            {
                s.resize(read);
            }

            return s;
        }

        /**
         * @brief Loads the data from an opened file at the current read position
         *
         * @param fp Pointer to opened file
         * @param fileSize File size
         * @param dst Destination string
         * @return size_t
         */
        static size_t loadFile(FILE *fp, size_t fileSize, string &dst)
        {
            // Resize the string to hold the entire file
            dst.resize(fileSize);
            // Read the whole file into dst and return the total of bu
            size_t nRead = std::fread(&dst[0], sizeof(char), fileSize, fp);

            // Fit the string to the data actually read
            if (nRead > 0 && nRead < fileSize)
            {
                dst.resize(nRead);
                // Shrink (may release memory or not, depends on implementation)
                dst.shrink_to_fit();
            }

            return dst.size();

            // Do not close fp, caller must take care of it
        }

        /**
         * @brief Splits a string by newline character
         *
         * @param str String to split
         * @return vector<string> Vector of substrings
         */
        static vector<string> splitLines(const string &str)
        {

            vector<string> lines;

            bool insideLine = false;
            bool isNewLine;

            size_t start = 0;
            size_t end{};
            size_t len = str.length();

            for (size_t p = start; p < len; p++)
            {
                const char c = str[p];

                isNewLine = false;
                // Check if current char is a newline
                if (c == '\r' || c == '\n')
                {
                    isNewLine = true;
                }

                // If current char is newline, check for line before this char
                if (isNewLine)
                {
                    if (insideLine && end >= start)
                    {
                        string token(str, start, end - start + 1);

                        lines.push_back(token);
                        insideLine = false;
                    }
                }
                else
                {
                    // If insideToken is false, start a new line
                    if (!insideLine)
                    {
                        // Mark start of line
                        start = p;
                    }
                    // Always mark the end.
                    end = p;
                    // Mark we're inside a token
                    insideLine = true;
                }
            }

            // Check for remaining data at the end
            if (insideLine)
            {
                string token(str, start, end - start + 1);
                lines.push_back(token);
                insideLine = false;
            }
            return lines;
        }

        /**
         * @brief Checks if a string has matching characters
         *
         * @param str String to check
         * @param opening Opening character
         * @param closing Closing character
         * @return true if opening and closing character match
         * @return false opening and closing character don't match
         */
        static bool matches(const string &str, char opening, char closing)
        {
            int count{};

            // Check if opening and closing char are equal
            if (opening == closing)
            {
                return false;
            }

            string::const_iterator iter;

            // Look for open char
            for (iter = str.begin(); iter != str.end(); iter++)
            {
                const char c = *iter;
                // Closing without opening
                if (c == closing && count == 0)
                {
                    return false;
                }

                if (c == opening)
                {
                    count++;
                }
                else if (c == closing)
                {
                    count--;
                }
            }

            return (count == 0);
        }
    };

    /** @brief Supported formats */
    enum class GridFormat : int
    {
        ESRI_ASCII,
        ESRI_FLOAT,
        ENVI_FLOAT,
        ENVI_DOUBLE,
        SURFER_ASCII,
        SURFER_FLOAT,
        SURFER_DOUBLE,
        TEXT,
        TEXT_REVERSE,
        UNKNOWN
    };

    /**
     * @brief Supported grid formats
     */
    static map<string, GridFormat> Formats{
        {"esriascii", GridFormat::ESRI_ASCII},
        {"esri", GridFormat::ESRI_FLOAT},
        {"envi", GridFormat::ENVI_FLOAT},
        {"envidouble", GridFormat::ENVI_DOUBLE},
        {"surferascii", GridFormat::SURFER_ASCII},
        {"surfer6", GridFormat::SURFER_FLOAT},
        {"surfer7", GridFormat::SURFER_DOUBLE},
        {"txt", GridFormat::TEXT},
        {"txtreverse", GridFormat::TEXT_REVERSE}};

    /**
     * @brief Get the format from a string
     *
     * @param formatString One of the supported grid formats
     * @return GridFormat, GridFormat::UNKNOWN if string is not one of the supported formats.
     */
    static inline GridFormat getFormat(string formatString)
    {
        GridFormat format = GridFormat::UNKNOWN;

        const string str = Strings::tolower(formatString);

        auto it = Formats.find(str);

        if (it != Formats.end())
        {
            return Formats[str];
        }

        return format;
    }

    /**
     * @brief Status of the operation.
     */
    enum class geoStatus : int
    {
        SUCCESS = 0,  /*!< OK */
        FAILURE = -1, /*!< Operation was not successful. */
    };

    /**
     * @brief Enables or disables debug messages
     * @param debugEnabled When true, enable debug messages
     */
    static inline void setDebug(bool debugEnabled)
    {
        geoDebug = debugEnabled;
    }

    template <typename Func>
    /**
     * Executes a function when debug is enabled
     */
    static auto ifDebug(Func f)
    {
        if (geoDebug)
        {
            f();
        }
    }

    /**
     * @brief Get page size from the operating system
     * @return Page size in bytes
     * @see https://create.stephan-brumme.com/portable-memory-mapping/
     */
    static inline int getPageSize()
    {
#ifdef _MSC_VER
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return sysInfo.dwAllocationGranularity;
#else
        return sysconf(_SC_PAGESIZE);
#endif
    }

    template <class T>
    /**
     * @brief Swaps two memory regions of the same size
     *
     * @param a Start of first memory region
     * @param b Start of second memory region
     * @param size Region size in bytes
     */
    static void memSwap(T *a, T *b, size_t size)
    {
        // Use the longest available data type
        long long *lA = reinterpret_cast<long long *>(a);
        long long *lB = reinterpret_cast<long long *>(b);

        // Calculate long chunk count
        size_t longCount = size / sizeof(long long);
        // Remaining memory needs to be copied byte by byte
        size_t byteCount = size % sizeof(long long);

        char *bA = reinterpret_cast<char *>(a);
        char *bB = reinterpret_cast<char *>(b);

        // Point to end of long long copy memory
        bA += longCount * sizeof(long long);
        bB += longCount * sizeof(long long);

        while (longCount-- > 0)
        {
            std::swap(*lA++, *lB++);
        }

        while (byteCount-- > 0)
        {
            std::swap(*bA++, *bB++);
        }
    }

    /**
     * @brief Returns the linear position for an element inside a 2D array, Row Major
     * @param row Row
     * @param column Column
     * @param columns Total of columns
     * @return Linear position
     */
    static inline int linear2D(int row, int column, int columns)
    {
        return (row * columns) + column;
    }

    /**
     * @brief Degrees to radians
     * @param d Degrees
     * @return Radians
     */
    static inline double radians(double d)
    {
        return ((d) * (pi / 180.0f));
    }

    /**
     * @brief Radians to degrees
     *
     * @param r
     * @return degrees
     */
    static inline double degrees(double r)
    {
        return (r * 180.0f) / pi;
    }

    /**
     * @brief Normalizes radians between -pi/pi
     *
     * @param r Radians
     * @return Normalized radians
     */
    static inline double normalizeRadians(double r)
    {

        while (r > pi)
        {
            r -= 2 * pi;
        }
        while (r < -pi)
        {
            r += 2 * pi;
        }
        return r;
    }

    /**
     * @brief Calculates target coordinates from source longitude, latitude, distance and bearing
     *
     * @param lon Starting longitude
     * @param lat Starting latitude
     * @param distance Distance (meters)
     * @param bearing Bearing angle, from north (0 degrees) clockwise
     * @return std::tuple<float, float> lon, lat target coordinates
     * @see https://www.movable-type.co.uk/scripts/latlong.html
     */
    static inline std::tuple<double, double> targetCoordinates(double lon, double lat, double distance, double bearing)
    {

        double radius{earthRadius};
        double angularDistance = (distance) / radius;
        double bearingRad = radians(bearing);
        double radLat1 = radians(lat);
        double radLon1 = radians(lon);

        double radLat2 = asinf(sinf(radLat1) * cosf(angularDistance) +
                               cosf(radLat1) * sinf(angularDistance) * cosf(bearingRad));

        double radLon2 = radLon1 + atan2f(sinf(bearingRad) * sinf(angularDistance) * cosf(radLat1),
                                          cosf(angularDistance) - sinf(radLat1) * sinf(radLat2));

        radLon2 = normalizeRadians(radLon2);

        return {degrees(radLon2), degrees(radLat2)};
    }

    /**
     * @brief Distance of 1 Arc Second (longitude, latitude) using the WGS84 Ellipsoid model
     * @see https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84
     * @param lat Latitude where the arc second distance is calculated
     * @return std::tuple<float, float> 1 arc distance in meters (longitude, latitude) for 1 arc sec at the given latitude
     */
    static inline std::tuple<double, double> arcSecMeters(double lat)
    {
        double a{earthRadius};              /*!< Semi-major axis (a) of the Earth (m) */
        double f{earthFlattening};          /*!< Flattening factor of the Earth */
        double e2 = {(2.0f * f) - (f * f)}; /*!< Eccentricity squared of the earth's ellipsoid: (2f - f^2)*/

        // Convert latitude to radians
        double latR = radians(lat);

        double sn2 = sinf(latR) * sinf(latR);
        double cs = cosf(latR);

        // Convert 1 arcsec to radians and multiply by the calculated distance
        double arcSecLon = radians(1.0f / 3600.0f) * ((a * cs) / powf(1.0 - e2 * sn2, 1.5f));
        double arcSecLat = radians(1.0f / 3600.0f) * ((a * (1 - e2)) / powf(1.0 - e2 * sn2, 1.5f));

        return {arcSecLon, arcSecLat};
    }

    /**
     * @brief Calculates the grid cell size in decimal degrees.
     * @param lat Latitude where the resolution is calculated.
     * @param dxM Grid X resolution in meters
     * @param dyM  Grid Y resolution in meters
     * @return std::tuple<double, double> X, Y resolution in decimal degrees
     */
    static inline std::tuple<double, double> cellSizeDegrees(double lat, double dxM, double dyM)
    {

        if (dxM == 0.0f || dyM == 0.0f)
        {
            return {0.0f, 0.0f};
        }

        // Get the distance of 1 arc second in meters at lat
        auto [arcSecLon, arcSecLat] = arcSecMeters(lat);

        // Divide dxM and dyM by the calculated distance of 1 arc second to calculate arcseconds
        // Divide resulting arc seconds by 3600 to convert to decimal degrees
        double dxDeg = (dxM / arcSecLon) / 3600.0f;
        double dyDeg = (dyM / arcSecLat) / 3600.0f;

        return {dxDeg, dyDeg};
    }

    /**
     * @brief Save projection file
     * @param path Path to grid file
     */
    static geoStatus saveWGS84Projection(const char *path)
    {

        fs::path filePath = fs::weakly_canonical(fs::path(string(path)));

        // If grid file does not exist, don't create projection file
        if (!fs::exists(filePath))
        {
            return geoStatus::FAILURE;
        }

        // Projection file has .prj extension
        filePath.replace_extension(".prj");

        ofstream ofs(filePath.string().c_str());

        // Always Geographic/WGS84
        ofs << "Projection         GEOGRAPHIC" << endl;
        ofs << "Datum              WGS84" << endl;
        ofs << "Spheroid           WGS84" << endl;
        ofs << "Units              DD" << endl;
        ofs << "Zunits             NO" << endl;
        ofs << "Parameters         /" << endl;

        if (!ofs.is_open())
        {
            return geoStatus::FAILURE;
        }

        return geoStatus::SUCCESS;
    }

    /**
     * @brief Options string/file parser
     * Parses options key EQUALSIGN value
     */
    class Options
    {

    public:
        /**
         * @brief Construct a new Options object
         *
         * @param text Options text. Several lines are separated by "\n"
         * @param equalSign Equal sign, defaults to '='
         */
        Options(const string &text = "", const char equalSign = '=') : equalSign(equalSign)
        {
            if (text.length())
            {
                string data(text);
                parse(data, this->equalSign);
            }
        }

        /**
         * @brief Sets the value for an option
         *
         * @param key Option key
         * @param value Option value
         */
        void set(const std::string &key, const std::string &value)
        {
            if (std::find_if(keys.begin(), keys.end(), [&key](const std::string &val)
                             { return (val.compare(key) == 0); }) == keys.end())
            {
                keys.push_back(key);
            }

            options[key] = value;
        }

        /**
         * @brief Parses and sets the value for an option
         *
         * @param option  string of type key EQUALSIGN value
         */
        void set(const std::string &option)
        {
            auto pos = option.find(this->equalSign);

            if (pos != string::npos)
            {
                string key(option, 0, pos);
                string value(option, pos + 1, option.length() - pos - 1);

                Strings::trim(key);
                Strings::trim(value);

                if (key.length())
                {
                    set(key, value);
                }
            }
        }

        /**
         * @brief Parses an option string. Multiple lines must be separated with '\n'
         *
         * @param data Data string to parse
         * @param equalSign Equal sign
         */
        virtual void parse(const string &data, char equalSign)
        {
            // Erase any existing data
            this->clear();

            // Get lines
            auto lines = Strings::splitLines(data);

            std::string currentOption = "";

            for (auto &l : lines)
            {
                Strings::trim(l);
                // Ignore blank and commented lines
                if (l.length() == 0 || l[0] == ';' || l[0] == '#')
                {
                    continue;
                }

                // Is there a new option?
                if (l.find_first_of(this->equalSign) != string::npos)
                {
                    if (currentOption.length())
                    {
                        // Parse current option
                        set(currentOption);
                        currentOption = Strings::trim(l);
                    }
                    else
                    {
                        currentOption += Strings::trim(l);
                    }
                }
                else
                {
                    currentOption += Strings::trim(l);
                }
            }

            // Parse last option
            if (currentOption.length())
            {
                set(currentOption);
            }
        }

        /**
         * @brief Creates a string representation of this instance
         *
         * @param glue Glue for each key value pair, defaults to "\n"
         * @return string
         */
        string str(const string &glue = "\n")
        {
            std::ostringstream oss;
            size_t n = keys.size();

            for (size_t i = 0; i < n; i++)
            {
                const string &key = keys[i];
                oss << key << equalSign << options[key];
                if (i < n - 1)
                {
                    oss << glue;
                }
            }

            return oss.str();
        }

        /**
         * @brief Clears this instance
         *
         */
        void clear()
        {
            this->options.clear();
            this->keys.clear();
        }

        /**
         * @brief Checks if a key exists
         * @param key Key to check
         */
        inline bool contains(const string &key)
        {
            return (options.find(key) != options.end());
        }

        /**
         * @brief Gets the value associated to a key
         *
         * @param key Key
         * @return string Value associated with the key
         */
        virtual string get(const string &key)
        {
            if (contains(key))
            {
                return options[key];
            }
            return "";
        }

        /**
         * @brief Checks if there are no options defined
         *
         * @return true if the options map is empty
         * @return false if there is at least one option defined
         */
        bool empty()
        {
            return (!keys.size());
        }

        /**
         * @brief Gets a float value
         *
         * @param name Key name
         * @return float Value
         */
        float getFloat(const string &name)
        {
            float value{};
            string s = get(name);
            if (s.size() > 0)
            {
                value = std::stof(s);
            }
            return value;
        }

        /**
         * @brief Gets an integer value
         *
         * @param name Key name
         * @return int Int value
         */
        int getInt(const string &name)
        {
            int value{};
            string s = get(name);
            if (s.size() > 0)
            {
                value = std::stoi(s);
            }
            return value;
        }

        /**
         * @brief Gets a list of string values
         *
         * @param name Key n ame
         * @param delim Value delimiter
         * @return vector<string> List of string values
         */
        virtual vector<string> getStrings(const string &name, const char &delim = ',')
        {
            vector<string> v;
            string s = get(name);
            if (s.size() > 0)
            {
                auto tokens = Strings::split(s, delim);
                for (auto &token : tokens)
                {
                    Strings::trim(token);
                    v.push_back(token);
                }
            }
            return v;
        }

        /**
         * @brief Gets a vector of floats
         * @param name Key name
         * @param delim Value delimiter
         * @return vector<float> Vector of floats
         */
        virtual vector<float> getFloats(const std::string &name, const char delim = ',')
        {
            std::vector<float> v;
            std::string s = get(name);
            if (s.size() > 0)
            {
                std::vector<std::string> vs = Strings::split(s, delim);
                for (int i = 0; i < vs.size(); i++)
                {
                    v.push_back(std::stof(vs[i]));
                }
            }
            return v;
        }

        /**
         * @brief Gets a vector of doubles
         * @param name Key
         * @param delim Value delimiter
         * @return vector<float> Vector of doubles
         */
        virtual vector<double> getDoubles(const std::string &name, const char delim = ',')
        {
            std::vector<double> v;
            std::string s = get(name);
            if (s.size() > 0)
            {
                std::vector<std::string> vs = Strings::split(s, delim);
                for (int i = 0; i < vs.size(); i++)
                {
                    v.push_back(std::stod(vs[i]));
                }
            }
            return v;
        }

        /**
         * @brief Gets a vector of integers
         * @param name Key name
         * @param delim Delimiter
         * @return vector<int> Vector of integers
         */
        virtual std::vector<int> getInts(const std::string &name, char delim = ',')
        {
            vector<int> v;
            string s = get(name);
            if (s.size() > 0)
            {
                vector<string> vs = Strings::split(s, delim);
                for (int i = 0; i < vs.size(); i++)
                {
                    v.push_back(std::stoi(vs[i]));
                }
            }
            return v;
        }

        /**
         * @brief Gets the keys
         * @return vector<string> Copy of the keys
         */
        std::vector<std::string> getKeys()
        {
            return keys;
        }

        /**
         * @brief Gets a copy of the values
         * @return vector<string> Copy of the values
         */
        std::vector<std::string> getValues()
        {
            std::vector<std::string> ret;

            for (auto it = options.begin(); it != options.end(); ++it)
            {
                ret.push_back(it->second);
            }
            return ret;
        }

        /**
         * @brief Gets a copy of the options map
         * @return map<string, string> Copy of options
         */
        std::map<std::string, std::string> all()
        {
            return options;
        }

        /**
         * @brief Checks if this object is valid
         *
         * @return true if it contains at least one key
         * @return false if it is empty
         */
        virtual bool valid()
        {
            return !empty();
        }

    protected:
        std::map<std::string, std::string> options; /*!< Options map */
        std::vector<std::string> keys;              /*!< Keys in order of insertion */
        char equalSign{'='};                        /*!< Equal sign*/
    };

    template <class T, class F>
    inline std::pair<const std::type_index, std::function<void(void *, char *, char **)>>
        /**
         * @brief Generic data parser
         * @param f Function to the associated type
         * @return Parser pair to insert into the parser map
         */
        static typeParser(const F &f)
    {
        return {
            std::type_index(typeid(T)),
            [g = f](void *a, char *pos, char **end)
            {
                if constexpr (std::is_void_v<T>)
                    g(pos, end);
                else
                    g(a, pos, end);
            }};
    }

    /**
     * @brief Parser map
     */
    static inline std::unordered_map<std::type_index, std::function<void(void *, char *, char **)>>
        parsers{
            typeParser<void>([](char *pos, char **end) {}),
            typeParser<int>([](void *ptr, char *pos, char **end)
                            {
													int * x = reinterpret_cast<int*>(ptr);
													*x = static_cast<int>(std::strtof(pos, end)); }),
            typeParser<unsigned>([](void *ptr, char *pos, char **end)
                                 {
															unsigned * x = reinterpret_cast<unsigned*>(ptr);
															*x = static_cast<unsigned>(std::strtof(pos, end)); }),
            typeParser<long>([](void *ptr, char *pos, char **end)
                             {
															long * x = reinterpret_cast<long*>(ptr);
															*x = static_cast<long>(std::strtof(pos, end)); }),
            typeParser<long long>([](void *ptr, char *pos, char **end)
                                  {
															long long * x = reinterpret_cast<long long*>(ptr);
															*x = static_cast<long long>(std::strtold(pos, end)); }),
            typeParser<float>([](void *ptr, char *pos, char **end)
                              {
														float * x = reinterpret_cast<float*>(ptr);
														*x = std::strtof(pos, end); }),
            typeParser<double>([](void *ptr, char *pos, char **end)
                               {
														double * x = reinterpret_cast<double*>(ptr);
														*x = std::strtod(pos, end); }),
            typeParser<long double>([](void *ptr, char *pos, char **end)
                                    {
														long double * x = reinterpret_cast<long double*>(ptr);
														*x = std::strtold(pos, end); }),
        };

    template <class T>
    static inline void parseType(T *ptr, char *pos, char **end)
    {
        if (const auto it = parsers.find(std::type_index(typeid(T)));
            it != parsers.cend())
        {
            it->second(ptr, pos, end);
        }
        else
        {
            ifDebug([&]
                    { cerr << "Unregistered type " << std::quoted(typeid(T).name()); });
        }
    }

    template <class T, class F>
    static inline void register_parser(const F &f)
    {
        if (geoDebug)
        {
            std::cout << "Register parser for type "
                      << std::quoted(typeid(T).name()) << '\n';
        }
        parsers.insert(typeParser<T>(f));
    }

    /**
     * @brief Text data
     * Loads a text file containing text representation of float items.
     * If a cache file exists for the data, attempts to load data from cache to avoid text parsing.
     */
    template <class T>
    struct DataSet
    {

        /**
         * @brief Loads a binary dataset
         *
         * @param path Path of the dataset file
         * @return tuple<status, size_t, T *>
         */
        static tuple<geoStatus, size_t, T *> loadBinary(string path)
        {
            if (!path.length())
            {
                return {geoStatus::FAILURE, 0, nullptr};
            }

            if (!fs::exists(path))
            {
                return {geoStatus::FAILURE, 0, nullptr};
            }

            fs::path binaryPath(path);

            auto binarySize = fs::file_size(binaryPath);

            // Calculate how many items are stored
            size_t binaryItems = binarySize / sizeof(T);

            if (binaryItems > 0)
            {
                // Allocate memory for all the items, plus one null item at the end
                T *binaryData = (T *)malloc((binaryItems + 1) * sizeof(T));
                // Unable to allocate memory?
                if (binaryData == nullptr)
                {
                    return {geoStatus::FAILURE, 0, nullptr};
                }
                FILE *binFile = fopen(binaryPath.string().c_str(), "rb");
                if (binFile != NULL)
                {

                    size_t readItems = fread(binaryData, sizeof(T), binaryItems, binFile);

                    fclose(binFile);

                    return {geoStatus::SUCCESS, readItems, binaryData};
                }
            }
            return {geoStatus::FAILURE, 0, nullptr};
        }

        /**
         * @brief Loads a binary dataset from the current file pointer read position
         *
         * @param fp File pointer, opened for reading and positioned at the start of data
         * @param fileSize Estimated file size
         * @return tuple<status, size_t, T *>  status, count and pointer to read data
         */
        static tuple<geoStatus, size_t, T *> loadBinary(FILE *fp, size_t fileSize)
        {
            if (fp == NULL)
            {
                return {geoStatus::FAILURE, 0, nullptr};
            }
            // Calculate how many items are stored
            size_t binaryItems = fileSize / sizeof(T);

            if (binaryItems > 0)
            {
                // Allocate memory for all the items, plus one null item at the end
                T *binaryData = (T *)malloc((binaryItems + 1) * sizeof(T));
                // Unable to allocate memory?
                if (binaryData == nullptr)
                {
                    return {geoStatus::FAILURE, 0, nullptr};
                }

                size_t readItems = fread(binaryData, sizeof(T), binaryItems, fp);

                binaryData = (T *)realloc(binaryData, readItems * sizeof(T));

                if (binaryData != nullptr)
                {

                    return {geoStatus::SUCCESS, readItems, binaryData};
                }
            }
            return {geoStatus::FAILURE, 0, nullptr};
        }

        /**
         * @brief Loads an array of T data from a text file
         *
         * @param fp Pointer to the opened file, loading starts at current position
         * @param fileSize Estimated file size
         * @return tuple<size_t, T*> count and array of T data <0, nullptr> if no data was read.
         */
        static tuple<geoStatus, size_t, T *> loadText(FILE *fp, size_t fileSize)
        {

            // Parser function
            std::function<void(void *, char *, char **)> parser;

            // Get this type size
            size_t typeSize = sizeof(T);

            // Locate parser for this type
            if (const auto it = parsers.find(std::type_index(typeid(T)));
                it != parsers.cend())
            {
                parser = it->second;
            }
            else
            {
                if (geoDebug)
                {
                    string errorMessage = string("Unregistered type ") + typeid(T).name();
                    std::cout << errorMessage << endl;
                }
                return {geoStatus::FAILURE, 0, nullptr};
            }

            // Get total char data length and pointer to the char data buffer
            auto [readDataStatus, charDataLength, charData] = readData(fp, fileSize);

            // Check if char data was not loaded
            if (readDataStatus != geoStatus::SUCCESS || charDataLength == 0)
            {
                return {geoStatus::FAILURE, 0, nullptr};
            }

            // Pointer to the start of the data buffer
            char *startData = charData;

            // Position to store the next T item in place (start of data buffer)
            T *currentItem = (T *)startData;

            // Point to start of char data
            char *pos = startData;

            size_t remaining = charDataLength;

            // Pointer to next position for data parsing
            char *end{};

            int count = 0;

            // Max item size in text representation
            size_t maxItemSize = 0;

            // While there are remaining characters, parse the next item
            T value;
            for (parser(&value, pos, &end); remaining > 0; parser(&value, pos, &end))
            {

                // Check data limits
                if (pos > charData + charDataLength)
                {
                    cerr << "Warning! attempting to parse outside of char data" << endl;
                    break;
                }

                if (end == NULL || end > charData + charDataLength)
                {
                    cerr << "Warning! last item spans outside of char data" << endl;
                    break;
                }

                if (pos == end)
                {
                    // If parsing did not succeed but there still are characters to process, print message skip one char on the text data
                    if (pos != NULL && pos < charData + charDataLength && !std::isspace(*pos) && *pos != 0)
                    {
                        // Invalid char?
                        // Skip!
                        pos++;
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }

                if (errno == ERANGE)
                {
                    errno = 0;
                    break;
                };

                // pos != end, there is data to store
                // Get this item text size
                size_t itemSize = end - pos;
                if (itemSize > maxItemSize)
                {
                    maxItemSize = itemSize;
                }

                // Attempt to store item in place, without overwriting remaining text data
                // Check if writing this item in place overwrites current parsing position
                if (((char *)currentItem + typeSize >= end))
                {

                    // Estimate how many items remain according to this item size
                    size_t remainingItems = remaining / std::min(itemSize, maxItemSize);

                    // Estimate total of items to store (current count + estimated remaining)
                    size_t totalItems = count + remainingItems;

                    // Estimate new array size
                    size_t estimatedSize = (totalItems)*typeSize;

                    // Reallocate the whole memory block
                    // Save current char data offset
                    // size_t charDataOffset = charData - startData;

                    // Offset of current position, pos and end
                    size_t currentItemOffset = (char *)currentItem - startData;
                    size_t posOffset = pos - charData;
                    size_t endOffset = end - pos;

                    // Allocate new memory block, plus one T at the end to fill with zeroes
                    startData = (char *)realloc(startData, estimatedSize + typeSize);

                    if (startData == NULL)
                    {
                        cerr << "Unable to relocate data" << endl;
                        return {geoStatus::FAILURE, 0, nullptr};
                    }

                    // Fill remaing data with spaces
                    memset(startData + estimatedSize, ' ', typeSize);
                    // Copy char data to new location!
                    std::copy(startData + posOffset, startData + posOffset + remaining, startData + estimatedSize - remaining);

                    // Update pointers
                    charData = startData + estimatedSize - remaining;
                    charDataLength = remaining;
                    pos = startData + estimatedSize - remaining + posOffset;
                    end = startData + estimatedSize - remaining + endOffset;
                    currentItem = (T *)(startData + currentItemOffset);
                }
                remaining -= end - pos;
                *currentItem = value;
                // Point to the next item position to insert a new item
                currentItem++;
                // Update current position
                pos = end;
                count++;
            }

            if (count)
            {
                // currentItem  points to one past the last item
                size_t realSize = (char *)currentItem - startData;
                // Resize to real contents
                // Reallocate memory to fit exactly realSize + 1 sentinel item filled with zeroes
                startData = (char *)realloc(startData, realSize + typeSize);

                if (startData == NULL)
                {
                    return {geoStatus::FAILURE, 0, nullptr};
                }
                else
                {

                    // Set the sentinel item past the last to the default value for the type
                    // if (debug) {
                    // cout <<"Finished reading " << count << " elements. Total size: " << realSize << endl;
                    //}
                    T sentinel{};
                    // currentItem points to past one the last element
                    *currentItem = sentinel;

                    return {geoStatus::SUCCESS, count, reinterpret_cast<T *>(startData)};
                }
            }
            else
            {
                // No items extracted?
                if (startData != NULL)
                {
                    // Force dispose.
                    free(startData);
                    return {geoStatus::FAILURE, 0, nullptr};
                }
            }
            return {geoStatus::FAILURE, 0, nullptr};
        }

        /**
         * @brief  Loads a text file into an array
         *
         * @param path File path
         * @return tuple<size_t, T *> size and pointer to array
         */
        static tuple<geoStatus, size_t, T *> loadText(string path)
        {
            if (!fs::exists(path))
            {
                return {geoStatus::FAILURE, 0, nullptr};
            }

            // Get type name
            string typeName = typeid(T).name();

            // Get file path
            fs::path filePath(path);

            size_t fileSize = fs::file_size(filePath);

            FILE *fp;
#ifdef _WIN32
            fp = fopen(path.c_str(), "rS"); // Read-only, optimized sequential access cache (mmap??)
#elif __linux__
            fp = fopen(path.c_str(), "rm"); // Use mmap (faster?)
#else
            fp = fopen(path.c_str(), "r"); // Plain read-only flags
#endif

            if (fp == NULL)
            {
                cerr << "Unable to open " << path << " for reading" << endl;
                return {geoStatus::FAILURE, 0, nullptr};
            }

            return loadText(fp, fileSize);
        }

        /**
         * @brief Load data from a file pointer
         * @param fp Pointer to a open file, reading position valid.
         * @param fileSize File size
         * @return tuple<status, size_t, char *> Operation result
         */
        static tuple<geoStatus, size_t, char *> readData(FILE *fp, size_t fileSize)
        {
            // Get system page size
            int pageSize = getPageSize();

            // Allocate memory for the whole file plus some bytes at the end
            char *data = (char *)malloc(fileSize + (fileSize % (2 * sizeof(T))));

            if (data == NULL)
            {
                return {geoStatus::FAILURE, 0, nullptr};
            }

            // Set two bytes to null AFTER file data in memory
            data[fileSize] = 0;
            data[fileSize + 1] = 0;

            // Read file chunks into memory: 16 pages each time
            size_t bufSize = 16 * pageSize;

            char *buf = data;

            if (bufSize > fileSize)
            {
                bufSize = fileSize;
            }

            size_t count = 0;
            size_t total = 0;

            int nReads = 0;

            size_t remaining = fileSize;

            while (!feof(fp))
            {
                count = fread(buf, sizeof(char), bufSize, fp);
                if (count > 0)
                {
                    nReads++;
                    total += count;
                    buf += count;
                    remaining -= count;
                    if (remaining == 0)
                    {
                        break;
                    }
                    if (bufSize > remaining)
                    {
                        bufSize = remaining;
                    }
                }
                else
                {
                    if (errno != EINVAL)
                    {
                        break;
                    }
                }
            }
            return {geoStatus::SUCCESS, total, data};
        }

        /**
         * @brief Writes a data array to a file
         * @param path Path to the ouput file
         * @param data Data array
         * @param count Array size
         * @param batchSize Insert a newline after writing this amount of items, 0 = single batch
         * @return status Operation status
         */
        static geoStatus saveText(string path, const T *data, size_t count, int batchSize)
        {
            // Open file in binary mode to avoid weird Windows file pointer mangling
            FILE *fp = fopen(path.c_str(), "wb");

            if (fp == nullptr)
            {
                return geoStatus::FAILURE;
            }

            geoStatus status = saveText(fp, 0, data, count, batchSize);

            fclose(fp);

            return status;
        }

        /**
         * @brief Saves a data array into a file at the starting offset
         *
         * @param fp File pointer opened for writing
         * @param offset Offset to relocate writing position, 0 does not relocate
         * @param data Data array
         * @param count Count of items
         * @param batchSize Insert a newline after writing this amount of items, 0 = single batch
         * @return status  Operation status
         */
        static geoStatus saveText(FILE *fp, size_t offset, const T *data, size_t count, int batchSize)
        {

            // Seek starting file position
            offset > 0 && fseek(fp, offset, SEEK_SET);

            int lastPercent = -1;

            size_t totalWritten = 0;

            if (batchSize == 0)
            {
                batchSize = count;
            }

            int totalBatches = count / batchSize;

            // Write data
            for (int i = 0; i < totalBatches; i++)
            {
                // cout << "Write line " << i * lineSize << " -> " << (i * lineSize) + lineSize << endl;
                for (int j = 0; j < batchSize && (i * batchSize) + j < count; j++)
                {
                    fprintf(fp, ((totalWritten % batchSize == 0) ? "%.7f" : " %.7f"), data[(i * batchSize) + j]);
                    totalWritten++;
                    int percent = floor(((float)totalWritten / (float)count) * 100.0f);

                    if (percent != lastPercent && percent % 10 == 0)
                    {
                        // Force data to be written
                        fflush(fp);

                        // ifDebug([&]
                        //         {
                        //          // Force cout to be written!
                        //      cout << (percent > 0 ? "..." : "") << percent;
                        //      cout.flush(); });

                        lastPercent = percent;
                    }
                }
                fprintf(fp, "\n");
            }

            // ifDebug([&]
            //         {
            //                  cout <<endl;
            //                  cout.flush(); });

            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a data array into a file at the starting offset
         *
         * @param fp File pointer opened for writing
         * @param offset Offset to relocate writing position, 0 does not relocate
         * @param data Data array
         * @param count Count of items
         * @param batchSize Count of items to write on each batch
         * @return status  Operation status
         */
        static geoStatus saveBinary(FILE *fp, size_t offset, const T *data, size_t count, int batchSize)
        {

            // Seek starting file position
            offset > 0 && fseek(fp, offset, SEEK_SET);

            int lastPercent = -1;

            size_t totalWritten = 0;

            if (batchSize == 0)
            {
                batchSize = count;
            }

            int totalBatches = count / batchSize;

            // Write data on batches
            for (int i = 0; i < totalBatches; i++)
            {
                size_t startPos = i * batchSize;
                size_t endPos = startPos + batchSize;
                if (endPos > count)
                {
                    endPos = count;
                }
                size_t batchLen = endPos - startPos;

                totalWritten += fwrite(&data[startPos], sizeof(T), batchLen, fp);
                int percent = floor(((float)totalWritten / (float)count) * 100.0f);

                if (percent != lastPercent && percent % 10 == 0)
                {
                    // Force data to be written
                    fflush(fp);
                    lastPercent = percent;
                }
            }

            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a data array into a file at the starting offset, reversing batches
         *
         * @param fp File pointer opened for writing
         * @param offset Offset to relocate writing position, 0 does not relocate
         * @param data Data array
         * @param count Count of items
         * @param batchSize Count of items to write on each batch
         * @return status  Operation status
         */
        static geoStatus saveBinaryReverse(FILE *fp, size_t offset, const T *data, size_t count, int batchSize)
        {

            // Seek starting file position
            offset > 0 && fseek(fp, offset, SEEK_SET);

            int lastPercent = -1;

            size_t totalWritten = 0;

            if (batchSize == 0)
            {
                batchSize = count;
            }

            int totalBatches = count / batchSize;

            // Write data on batches
            for (int i = totalBatches - 1; i >= 0; i--)
            {
                size_t startPos = i * batchSize;
                size_t endPos = startPos + batchSize;
                if (endPos > count)
                {
                    endPos = count;
                }
                size_t batchLen = endPos - startPos;

                totalWritten += fwrite(&data[startPos], sizeof(T), batchLen, fp);
                int percent = floor(((float)totalWritten / (float)count) * 100.0f);

                if (percent != lastPercent && percent % 10 == 0)
                {
                    // Force data to be written
                    fflush(fp);

                    // ifDebug([&]
                    //         {
                    //              // Force cout to be written!
                    //          cout << (percent > 0 ? "..." : "") << percent;
                    //          cout.flush(); });

                    lastPercent = percent;
                }
            }

            // ifDebug([&]
            //         {
            //                  cout <<endl;
            //                  cout.flush(); });

            return geoStatus::SUCCESS;
        }

        /**
         * @brief Writes a data array to a file
         * @param path Path to the ouput file
         * @param data Data array
         * @param count Array size
         * @param batchSize Insert a newline after writing this amount of items, 0 = single line
         * @return status Operation status
         */
        static geoStatus saveTextReverseBatches(string path, const T *data, size_t count, int batchSize)
        {
            // Open file in binary mode to avoid weird Windows file pointer mangling
            FILE *fp = fopen(path.c_str(), "wb");

            if (fp == nullptr)
            {
                return geoStatus::FAILURE;
            }

            geoStatus status = saveTextReverseBatches(fp, 0, data, count, batchSize);

            fclose(fp);

            return status;
        }

        /**
         * @brief Saves a data array into a file at the starting offset
         *
         * @param fp File pointer opened for writing
         * @param offset Offset to relocate writing position, 0 does not relocate
         * @param data Data array
         * @param count Count of items
         * @param batchSize Insert a newline after writing this amount of items, 0 = single line
         * @return status  Operation status
         */
        static geoStatus saveTextReverseBatches(FILE *fp, size_t offset, const T *data, size_t count, int batchSize)
        {

            // Seek starting file position
            offset > 0 && fseek(fp, offset, SEEK_SET);

            int lastPercent = -1;

            size_t totalWritten = 0;

            if (batchSize == 0)
            {
                batchSize = count;
            }

            int totalBatches = count / batchSize;

            for (int i = totalBatches - 1; i >= 0; i--)
            {
                for (int j = 0; j < batchSize && (i * batchSize) + j < count; j++)
                {
                    fprintf(fp, ((totalWritten % batchSize == 0) ? "%.7f" : " %.7f"), data[(i * batchSize) + j]);
                    totalWritten++;
                    int percent = floor(((float)totalWritten / (float)count) * 100.0f);

                    if (percent != lastPercent && percent % 10 == 0)
                    {
                        // Force data to be written
                        fflush(fp);

                        // ifDebug([&]
                        //         {
                        //          // Force cout to be written!
                        //      cout << (percent > 0 ? "..." : "") << percent;
                        //      cout.flush(); });

                        lastPercent = percent;
                    }
                }
                fprintf(fp, "\n");
            }

            // ifDebug([&]
            //         {
            //                  cout <<endl;
            //                  cout.flush(); });

            return geoStatus::SUCCESS;
        }
    }; // End struct DataSet

    /**
     * @brief 2D grid
     */
    class Grid
    {

    public:
        /**
         * @brief Construct a new empty grid
         */
        Grid()
        {
            /* Nothing to do, attribues already default initialized */
        }

        /**
         * @brief Construct a new Grid object with the specified dimensions
         * @param format Grid format
         * @param data Pointer to data array containing (row * column) elements
         * @param rows Grid rows - longitude
         * @param columns Grid columns - latitude
         * @param x0 Longitude of the lower left corner
         * @param y0 Latitude of the lower left corner
         * @param dx Point separation - longitude (meters)
         * @param dy Point separation - latitude (meters)
         * @param dxDeg Point separation l longitude in decimal degrees
         * @param dyDeg Point separation l longitude in decimal degrees
         * @param noData value
         */
        Grid(
            GridFormat format,
            float *data,
            int rows,
            int columns,
            double x0,
            double y0,
            double dx,
            double dy,
            double dxDeg,
            double dyDeg,
            float noData = NAN)
        {
            Grid::setup(format, *this, data, rows, columns, x0, y0, dx, dy, dxDeg, dyDeg, noData);
        }

        /**
         * @brief Construct a new Grid object with the specified dimensions
         * @param format Grid format
         * @param data Pointer to data array containing (row * column) elements
         * @param rows Grid rows - longitude
         * @param columns Grid columns - latitude
         * @param x0 Longitude of the lower left corner
         * @param y0 Latitude of the lower left corner
         * @param dx Point separation - longitude (meters)
         * @param dy Point separation - latitude (meters)
         * @param noData NODATA value
         */
        Grid(
            GridFormat format,
            float *data,
            int rows,
            int columns,
            double x0,
            double y0,
            double dx,
            double dy,
            float noData = NAN)
        {

            // Calculate dx,dy in decimal degrees from dx, dy in meters at the latitude of the grid origin
            auto [dxDeg, dyDeg] = cellSizeDegrees(y0, dx, dy);
            // Set this instance attributes
            Grid::setup(format, *this, data, rows, columns, x0, y0, dx, dy, dxDeg, dyDeg, noData);
        }

        /**
         * @brief Releases grid data
         */
        ~Grid()
        {
            dispose();
        }

        /**
         * @brief Copy constructor
         *
         * @param rhs Grid instance
         */
        Grid(const Grid &rhs)
        {
            copyFrom(rhs);
        }

        /**
         * @brief Copy assignment constructor
         * @param rhs Instance to be copied
         * @return Grid&
         */
        Grid &operator=(const Grid &rhs)
        {
            if (this != &rhs)
            {
                dispose();
                // Copy data from the rhs instance
                copyFrom(rhs);
            }
            return *this;
        }

        /**
         * @brief Move constructor
         *
         * @param rhs Instance to be moved into this instance
         */
        Grid(Grid &&rhs)
        {
            moveFrom(rhs);
        }

        /**
         * @brief Move asignment operator
         *
         * @param rhs Instance do be moved into this instance
         * @return Grid& This grid after moving
         */
        Grid &operator=(Grid &&rhs)
        {
            if (this != &rhs)
            {
                dispose();

                moveFrom(rhs);
            }
            return *this;
        }

        /**
         * @brief Initializes a grid with the given attributes
         * @param format Grid format
         * @param grid Reference to the grid
         * @param data Array of (rows * columns)
         * @param rows Count of cells in latitude direction
         * @param columns Count of cells in longitude direction
         * @param x0 Grid lower left longitude
         * @param y0 Grid lower left latitude
         * @param dx Cell size in the longitude direction (meters)
         * @param dy Cell size in the latitude direction (meters)
         * @param dxDeg Cell X size in degrees
         * @param dyDeg Cell Y size in degrees
         * @param noData Nodata value
         * @return Reference to the same initialized grid
         */
        static Grid &setup(GridFormat format,
                           Grid &grid,
                           float *data,
                           int rows,
                           int columns,
                           double x0,
                           double y0,
                           double dx,
                           double dy,
                           double dxDeg,
                           double dyDeg,
                           float noData = NAN)
        {
            grid.dispose();

            grid.rows = rows;
            grid.columns = columns;
            grid.x0 = x0;
            grid.y0 = y0;
            grid.dx = dx;
            grid.dy = dy;
            grid.dxDeg = dxDeg;
            grid.dyDeg = dyDeg;
            grid.data = data;
            grid.noData = noData;

            // Check if last position of data can be accessed.
            if (data != nullptr)
            {
                auto v = grid.data[(grid.rows * grid.columns) - 1];

                // Use value to keep the compiler happy
                std::swap(v, v);
            }

            return grid;
        }

        /**
         * @brief Returns the underlying data pointer
         * @return Pointer to data array
         */
        float *c_float() const
        {
            return this->data;
        }

        /**
         * @brief Returns the grid extents
         */
        std::tuple<double, double, double, double> extents() const
        {
            double x0 = this->x0;
            double y0 = this->y0;

            // target yMax: 0 degrees bearing (North), rows * dy in meters
            // auto [ignoreX, yMax] = targetCoordinates(x0, y0, this->rows * this->dy, 0);
            // target xMax: 90 degrees bearing (East), rows * dx in meters
            // auto[xMax, ignoreY] = targetCoordinates(x0, y0, this->columns * this->dx, 90);
            auto xMax = x0 + (this->dxDeg * this->columns);
            auto yMax = y0 + (this->dyDeg * this->rows);

            return {x0, y0, xMax, yMax};
        }

        /**
         * @brief Returns the resolution of the grid
         * @return {dx in degrees, dy in degrees}
         */
        std::tuple<double, double> resolution() const
        {
            return {this->dxDeg, this->dyDeg};
        }

        /**
         * @brief Returns the grid dimensions
         * @return {rows, columns}
         */
        std::tuple<int, int> dimensions() const
        {
            return {rows, columns};
        }

        /**
         * @brief Returns noData value
         * @return This instance noData value
         */
        double noDataValue() const
        {
            return this->noData;
        }

        /**
         * @brief Get a reference to the element at the specified position
         * @param row Row
         * @param column Column
         * @return float&
         */
        float &operator()(int row, int column)
        {
            return this->data[(row * columns) + column];
        }

        /**
         * @brief Get a reference to the element at the specified position
         * @param row Row
         * @param column Column
         * @return float&
         */
        float &operator()(int row, int column) const
        {
            return this->data[(row * columns) + column];
        }

        float equalsAt(const Grid &rhs, int row, int column, float rpe = 1.0f) const
        {
            const float valueA = (*this)(row, column);
            const float valueB = rhs(row, column);
            // Check if both values are nan, in such case they're considered equal.
            if (std::isnan(valueA) && std::isnan(valueB))
            {
                return true;
            }

            float err = fabs((valueB - valueA) / valueB) * 100.0f;

            return err < rpe;
        }

        /**
         * @brief Checks if this grid has equal dimensions with rhs
         * @param rhs Grid to compare
         * @return true if both grids have the same dimensions
         * @return false  when both grids have different dimensions
         */
        bool equalDimensions(const Grid &rhs) const
        {
            return (this->rows == rhs.rows && this->columns == rhs.columns);
        }

        /**
         * @brief Reverse rows inside the grid
         */
        void reverseRows()
        {
            if (data == NULL)
            {
                return;
            }
            for (int i = 0; i < rows / 2; i++)
            {
                // Reverse the entire row
                memSwap(data + (i * columns), data + ((rows - i - 1) * columns), columns * sizeof(float));
                /*
                // Reverse each element
                for (int j = 0; j < columns; j++)
                {
                  int pos = (i * columns) + j;
                  int newPos = ((rows - i - 1) * columns) + j;
                  std::swap(data[pos], data[newPos]);
                }
                  */
            }
        }

        /**
         * @brief Fills grid data with a value
         *
         * @param value
         */
        void fill(float value)
        {
            for (int i = 0; i < rows * columns; i++)
            {
                data[i] = value;
            }
        }

        /**
         * @brief Load data from a text file
         *
         * @param path Path to the text file
         * @param rows Rows
         * @param columns Columns
         * @param x0 Lower left corner X
         * @param y0 Lower left corner y
         * @param dx Grid x resolution in meters
         * @param dy Grid y resolution in meters
         * @param nodata Nodata value
         * @param reverseRows true if last row is the first line on the file
         * @return status Operation status
         */
        geoStatus loadText(
            string path,
            int rows,
            int columns,
            double x0,
            double y0,
            double dx,
            double dy,
            float nodata = NAN,
            bool reverseRows = false)
        {

            // Erase any previous data held on this instance
            this->dispose();

            if (!fs::exists(path) || !fs::is_regular_file(path))
            {
                return geoStatus::FAILURE;
            }

            size_t fileSize = fs::file_size(path);
            if (fileSize <= 0)
            {
                return geoStatus::FAILURE;
            }

            FILE *fp = fopen(path.c_str(), "r");

            if (fp == NULL)
            {
                return geoStatus::FAILURE;
            }

            return this->loadText(fp, fileSize, rows, columns, x0, y0, dx, dy, nodata, reverseRows);
        }

        /**
         * @brief Loads data from a TXT file
         *
         * @param fp Pointer to an opened file. Read position must be at the start of data
         * @param fileSize File size
         * @param rows Grid rows
         * @param columns Grid columns
         * @param x0 Lower left corner longitude
         * @param y0 Lower left corner latitude
         * @param dx Resolution in longitude
         * @param dy Resolution in latitude
         * @param noData NoData value
         * @param reverseRows true if the rows are reversed on the file
         * @return status status::SUCCESS if load was successful, status::FAILURE if load fails
         * @note File pointer is not closed.
         */
        geoStatus loadText(
            FILE *fp,
            size_t fileSize,
            int rows,
            int columns,
            double x0,
            double y0,
            double dx,
            double dy,
            float noData = NAN,
            bool reverseRows = false)
        {

            // Erase any previous data held on this instance
            this->dispose();

            this->rows = rows;
            this->columns = columns;
            this->x0 = x0;
            this->y0 = y0;
            this->dx = dx;
            this->dy = dy;

            // Calculate dx and dy in degrees
            auto [dDx, dDy] = cellSizeDegrees(y0, dx, dy);

            this->dxDeg = dDx;
            this->dyDeg = dDy;

            // fp points to the first row on the file
            // Load rows into the data buffer in place
            auto [status, count, data] = DataSet<float>::loadText(fp, fileSize);

            // Assign data
            this->data = data;

            // Check if the whole file was loaded
            // If not, discard partial loaded grid
            if (status != geoStatus::SUCCESS || count < this->rows * this->columns)
            {
                cerr
                    << "Warning! file should contain "
                    << rows * columns
                    << " values, but only "
                    << count << " values were read.\nReleasing partial data..." << endl;
                // Dispose read data
                this->dispose();
                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Reverse rows if required
            if (reverseRows)
            {
                this->reverseRows();
            }

            this->noData = noData;

            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a 2D grid into a TXT file (Row-Major)
         * @param path to save the file
         * @param reverseRows true to store last row first
         * @return operation status
         */
        geoStatus saveText(const char *path, bool reverseRows = false) const
        {

            int count = this->rows * this->columns;

            if (this->data == nullptr || count == 0)
            {
                // ifDebug([&]
                //         { cerr << "Grid is empty, nothing to save" << endl; });

                return geoStatus::FAILURE;
            }

            fs::path dataPath(path);

            if (reverseRows)
            {
                return DataSet<float>::saveTextReverseBatches(dataPath.string(), this->data, count, this->columns);
            }
            else
            {
                return DataSet<float>::saveText(dataPath.string(), this->data, count, this->columns);
            }
        }

    private:
        float *data{nullptr}; /*!< Flat array of data (row1row2row3...), no padding between rows*/
        int rows{};           /*!< Grid rows */
        int columns{};        /*!< Grid columns */
        double x0{};          /*!< X coordinate (longitude, decimal degrees) of the lower left corner */
        double y0{};          /*!< Y coordinate (latitude, decimal degrees) of the lower left corner */
        double dx{};          /*!< X resolution in meters */
        double dy{};          /*!< Y resolution in meters */
        double dxDeg{};       /*!< X resolution in decimal degrees */
        double dyDeg{};       /*!< Y resolution in decimal degrees */
        float noData{NAN};    /*!< NoData value */

        /**
         * @brief Copies data from another instance
         *
         * @param rhs Grid instance
         */
        void copyFrom(const Grid &rhs)
        {
            this->rows = rhs.rows;
            this->columns = rhs.columns;
            this->x0 = rhs.x0;
            this->y0 = rhs.y0;
            this->dx = rhs.dx;
            this->dy = rhs.dy;
            this->dxDeg = rhs.dxDeg;
            this->dyDeg = rhs.dyDeg;

            if (this->rows * this->columns <= 0)
            {
                return;
            }

            size_t memSize = this->rows * this->columns * sizeof(float);
            if (memSize > 0)
            {

                if (rhs.data != nullptr)
                {
                    // Copy data from rhs
                    this->data = (float *)malloc(memSize);
                    std::copy(rhs.data, rhs.data + memSize, this->data);
                }
            }
        }

        /**
         * @brief Moves data from other instance
         *
         * @param rhs Instance to move data. It becomes default initialized.
         */
        void moveFrom(Grid &rhs)
        {
            this->rows = rhs.rows;
            this->columns = rhs.columns;
            this->x0 = rhs.x0;
            this->y0 = rhs.y0;
            this->dx = rhs.dx;
            this->dy = rhs.dy;
            this->dxDeg = rhs.dxDeg;
            this->dyDeg = rhs.dyDeg;
            // Grab data pointer from rhs
            this->data = rhs.data;

            // Empty rhs
            rhs.rows = 0.0f;
            rhs.columns = 0.0f;
            rhs.x0 = 0.0f;
            rhs.y0 = 0.0f;
            rhs.dx = 0.0f;
            rhs.dy = 0.0f;
            rhs.dxDeg = 0.0f;
            rhs.dyDeg = 0.0f;
            rhs.data = nullptr;
        }

        /**
         * @brief Dispose this instance data
         *
         */
        void dispose()
        {

            if (this->data != nullptr)
            {
                free(this->data);
            }
            // Initialize all parameters
            this->rows = 0;
            this->columns = 0;
            this->x0 = 0.0f;
            this->y0 = 0.0f;
            this->dx = 0.0f;
            this->dy = 0.0f;
            this->dxDeg = 0.0f;
            this->dyDeg = 0.0f;
            this->data = nullptr;
        }
    }; // End class

    /**
     * @brief ESRI grids
     *
     */
    struct Esri
    {

        /** @brief NAN value */
        static constexpr auto nan = NAN;

        /**
         * @brief ESRI grid data types
         */
        enum class DataType
        {
            u8 = 1,   // Byte: 8-bit unsigned integer
            i16 = 2,  // Integer: 16-bit signed integer
            i32 = 3,  // Long: 32-bit signed integer
            f32 = 4,  // Floating-point: 32-bit single-precision
            d64 = 5,  // Double-precision: 64-bit double-precision floating-point
            ff32 = 6, // Complex: Real-imaginary pair of single-precision floating-point
            dd64 = 9, // Double-precision complex: Real-imaginary pair of double precision floating-point
            u16 = 12, // Unsigned integer: 16-bit
            u32 = 13, // Unsigned long integer: 32-bit
            i64 = 14, // 64-bit long integer (signed)
            u64 = 15  // 64-bit unsigned long integer (unsigned)
        };

        /**
         * @brief ESRI grid Byte order
         *
         */
        enum class ByteOrder
        {
            LE = 0, /*!< Little Endian*/
            BE = 1  /*!< Big Endian*/
        };

        /**
         * @brief ESRI ASCII header
         *
         */
        class Header : public Options
        {
        public:
            /**
             * @brief Construct a new Header object
             *
             * @param text Header text as string
             */
            Header(const string &text = "") : Options(text, ' ')
            {
            }

            /**
             * @brief Construct a new Header object from a file pointer
             *
             * @param fp Pointer to opened header file
             * @param fileSize File size in bytes
             */
            Header(FILE *fp, size_t fileSize) : Header("")
            {
                parseFile(fp, fileSize);
            }

            /**
             * @brief Parses header
             *
             * @param data Header data as string
             * @param equalSign Equal sign. Overriden to ' ' (space)
             */
            virtual void parse(const string &data, char equalSign = ' ') override
            {
                // Override user specified sign, ESRI uses ' ' as equal sign
                Options::parse(data, this->equalSign);

                // No key / values defined? return.
                if (this->empty())
                {
                    return;
                }

                if (this->contains("nrows") && this->contains("nrows"))
                {
                    this->nRows = this->getInt("nrows");
                    this->nCols = this->getInt("ncols");
                    if (this->nRows > 0 && this->nCols > 0)
                    {
                        this->dimensionDefined = true;
                    }
                    else
                    {
                        // rows or cols are zero
                        this->nRows = 0;
                        this->nCols = 0;
                    }
                }

                if (this->contains("dx") && this->contains("dy"))
                {
                    this->dx = this->getFloat("dx");
                    this->dy = this->getFloat("dy");
                    if (this->dx > 0.0f && this->dy > 0.0f)
                    {
                        this->resolutionDefined = true;
                    }
                    else
                    {
                        // dx or dy are zero
                        this->dx = 0.0f;
                        this->dy = 0.0f;
                    }
                }
                else if (this->contains("cellsize"))
                {
                    this->dy = this->dx = this->getFloat("cellsize");
                    if (this->dx > 0.0f && this->dy > 0.0f)
                    {
                        this->resolutionDefined = true;
                    }
                    else
                    {
                        // dx or dy are zero
                        this->dx = 0.0f;
                        this->dy = 0.0f;
                    }
                }

                if (this->contains("xllcorner") && this->contains("yllcorner"))
                {
                    this->cornerDefined = true;
                    this->xllCorner = this->getFloat("xllcorner");
                    this->yllCorner = this->getFloat("yllcorner");
                    // Calculate xllcenter and yllcenter
                    this->xllCenter = this->xllCorner + (this->dx / 2.0f);
                    this->xllCenter = this->yllCorner + (this->dy / 2.0f);
                    this->originDefined = true;
                }
                else if (this->contains("xllcenter") && this->contains("yllcenter"))
                {
                    this->cornerDefined = false;
                    this->xllCenter = this->getFloat("xllcorner");
                    this->xllCenter = this->getFloat("yllcorner");

                    // Calculate xllcorner and yllcorner
                    this->xllCorner = this->xllCenter - (this->dx / 2.0f);
                    this->xllCorner = this->yllCenter - (this->dy / 2.0f);
                    this->originDefined = true;
                }

                if (this->contains("nodata_value"))
                {
                    this->noData = this->getFloat("nodata_value");
                    this->noDataDefined = true;
                }
            }

            /**
             * @brief Parses the header from a file pointer
             *
             * @param fp Pointer to already opened file
             * @param fileSize file size
             */
            virtual void parseFile(FILE *fp, size_t fileSize)
            {

                // No valid file stream, return.
                if (fp == NULL)
                {
                    return;
                }

                // Header is always at the start of the file
                fseek(fp, 0L, SEEK_SET);

                string headerString; // Header string

                int count = 0; // Count of lines read so far
                int limit = 7; // Max. 7 lines

                do
                {
                    char buf[BUFSIZ];
                    if (fgets(buf, BUFSIZ, fp) == NULL)
                    {
                        break;
                    }
                    string line(buf);
                    headerString.append(Strings::tolower(Strings::trim(line)) + "\n");
                } while (!feof(fp) && ++count < limit && headerString.find("nodata_value") == string::npos);

                this->parse(headerString);
            }

            /**
             * @brief Checks if this object is valid
             *
             * @return true if it contains at least one key
             * @return false if it is empty
             */
            bool valid() override
            {
                if (!Options::valid())
                {
                    return false;
                }

                if (!this->dimensionDefined || !this->originDefined || !this->resolutionDefined || !this->noDataDefined)
                {
                    return false;
                }
                return true;
            }

            /**
             * @brief Get the Parameters object
             * @return tuple<int, int, double, double, double, double, float> rows, columns, x0, y0, dx, dy, noData
             */
            tuple<int, int, double, double, double, double, float> getParameters()
            {
                return {nRows, nCols, xllCorner, yllCorner, dx, dy, noData};
            }

        protected:
            int nRows{};                   /*!< Grid rows */
            int nCols{};                   /*!< Grid columns */
            double dx{};                   /*!< Resolution in X direction (longitude) - decimal degrees*/
            double dy{};                   /*!< Resolution in Y direction (latitude) - decimal degrees */
            double xllCorner{};            /*!< Lower left corner X */
            double xllCenter{};            /*!< Lower left corner Y */
            double yllCorner{};            /*!< Lower left cell center X*/
            double yllCenter{};            /*!< Lower left cell center Y */
            float noData{};                /*!< NODATA value */
            bool cornerDefined{false};     /*!< True when corners are defined */
            bool dimensionDefined{false};  /*!< True when dimension is defined */
            bool resolutionDefined{false}; /*!< True when resolution is defined */
            bool originDefined{false};     /*!< True when origin is defined */
            bool noDataDefined{false};     /*!< True when NODATA value is defined*/
        };

        /**
         * @brief Binary header
         *
         */
        class BinaryHeader : public Header
        {
        public:
            /**
             * @brief Construct a new Header object
             *
             * @param text Header text as string
             */
            BinaryHeader(const string &text = "") : Header(text)
            {
            }

            /**
             * @brief Construct a new Header object from a file pointer
             *
             * @param fp Pointer to opened header file
             * @param fileSize File size in bytes
             */
            BinaryHeader(FILE *fp, size_t fileSize) : BinaryHeader("")
            {
                this->parseFile(fp, fileSize);
            }

            /**
             * @brief Parses header
             *
             * @param data BinaryHeader data as string
             * @param equalSign Equal sign. Overriden to ' ' (space)
             */
            void parse(const string &data, char equalSign = ' ') override
            {
                // Override user specified sign, ESRI uses ' ' as equal sign
                Options::parse(data, this->equalSign);

                // No key / values defined? return.
                if (this->empty())
                {
                    return;
                }

                if (this->contains("nrows") && this->contains("ncols"))
                {
                    this->nRows = this->getInt("nrows");
                    this->nCols = this->getInt("ncols");
                    if (this->nRows > 0 && this->nCols > 0)
                    {
                        this->dimensionDefined = true;
                    }
                    else
                    {
                        // rows or cols are zero
                        this->nRows = 0;
                        this->nCols = 0;
                    }
                }

                // Parse dx, dy first

                if (this->contains("xdim") && this->contains("ydim"))
                {
                    this->dx = this->getFloat("xdim");
                    this->dy = this->getFloat("ydim");
                    if (this->dx > 0.0f && this->dy > 0.0f)
                    {
                        this->resolutionDefined = true;
                    }
                    else
                    {
                        // dx or dy are zero
                        this->dx = 0.0f;
                        this->dy = 0.0f;
                    }
                }
                else if (this->contains("cellsize"))
                {
                    this->dy = this->dx = this->getFloat("cellsize");
                    if (this->dx > 0.0f && this->dy > 0.0f)
                    {
                        this->resolutionDefined = true;
                    }
                    else
                    {
                        // dx or dy are zero
                        this->dx = 0.0f;
                        this->dy = 0.0f;
                    }
                }

                // Calculate x0, y0: requires dx and dy
                if (this->resolutionDefined && this->contains("ulxmap") && this->contains("ulymap"))
                {

                    // Calculate xllcorner and yllcorner
                    this->xllCenter = this->getFloat("ulxmap"); // Upper and lower x are the same
                    this->xllCorner = this->xllCenter - (this->dx / 2.0f);

                    // yllcorner:
                    this->yllCenter = this->getFloat("ulymap") - ((this->nRows - 1) * this->dy);
                    this->yllCorner = this->yllCenter - (this->dy / 2.0f);

                    this->cornerDefined = true;

                    this->originDefined = true;
                }

                if (this->contains("nodata"))
                {
                    this->noData = this->getFloat("nodata");
                    this->noDataDefined = true;
                }
            }

            /**
             * @brief Parses the header from a file pointer
             *
             * @param fp Pointer to already opened file
             * @param fileSize file size
             */
            void parseFile(FILE *fp, size_t fileSize) override
            {

                // No valid file stream, return.
                if (fp == NULL)
                {
                    return;
                }

                // Header is always at the start of the file
                fseek(fp, 0L, SEEK_SET);

                string headerString; // Header string

                // Binary header needs to read the whole file
                while (!feof(fp))
                {
                    char buf[BUFSIZ];
                    if (fgets(buf, BUFSIZ, fp) == NULL)
                    {
                        break;
                    }
                    string line(buf);
                    headerString.append(Strings::tolower(Strings::trim(line)) + "\n");
                }

                this->parse(headerString);
            }
        };

        /**
         * @brief Loads an ESRI ASCII into a grid instance
         * @param grid Target grid
         * @param path Path to the ESRI ASCII grid (.asc) file
         * @return status status::SUCCESS if load was successful, status::FAILURE if load fails
         */
        static geoStatus loadAscii(Grid &grid, const string &path)
        {

            fs::path p(path);

            // Check if file exists
            if (!fs::exists(p) || !fs::is_regular_file(p))
            {
                return geoStatus::FAILURE;
            }

            // Convert path to canonical
            p = fs::canonical(p);
            size_t fileSize = fs::file_size(p);

            FILE *fp = fopen(p.string().c_str(), "r");
            if (fp == nullptr)
            {
                cerr << "Unable top open " << path << endl;
                return geoStatus::FAILURE;
            }

            // Load header from file
            Header h(fp, fileSize);

            // Check if header is valid
            if (!h.valid())
            {
                // ifDebug([&]
                //         { cerr << path << " does not contain a valid header." << endl; });

                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Header is valid

            // Get parameters from the header
            auto [rows, columns, x0, y0, dxDeg, dyDeg, noData] = h.getParameters();

            // Get how many meters has 1 arcsec at y0 latitude
            auto [lonMeters, latMeters] = arcSecMeters(y0);

            // Calculate dx in meters: (round down)
            auto dx = floorf(
                dxDeg * 3600 // Convert dxDeg to arc secs
                * lonMeters  // Multiply by how many lon meters are there in 1 arcsec at this lat
            );

            // Calculate dy in meters: (round down)
            auto dy = floorf(
                dyDeg * 3600 // Convert dyDeg to arc secs
                * latMeters  // Multiply by how many lat meters are there in 1 arcsec at this lat
            );

            // fp points to the first row on the file
            // Load rows into the data buffer in place
            auto [status, count, data] = DataSet<float>::loadText(fp, fileSize);

            // Check if the whole file was loaded
            // If not, discard partial loaded grid
            if (status != geoStatus::SUCCESS || count < rows * columns)
            {
                // ifDebug([&]
                //         { cerr
                //               << "Warning! "
                //               << p.stem().string()
                //               << " should contain "
                //               << rows * columns
                //               << " values, but only "
                //               << count << " values were read.\nReleasing partial data..." << endl; });

                // Dispose partially read data
                free(data);
                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Setup grid
            Grid::setup(GridFormat::ESRI_ASCII, grid, data, rows, columns, x0, y0, dx, dy, dxDeg, dyDeg, noData);

            // ESRI ASCII stores last row at the top, rows need to be reversed.
            grid.reverseRows();

            // Close file pointer
            fclose(fp);
            return geoStatus::SUCCESS;
        }

        /**
         * @brief Loads an ESRI ASCII float binary grid
         * @param grid Target grid
         * @param path Path to the ESRI binary grid (.bil) file (only single-band bsq supported)
         * @return status status::SUCCESS if load was successful, status::FAILURE if load fails
         */
        static geoStatus loadFloat(Grid &grid, const string &path)
        {

            if (!path.length())
            {
                return geoStatus::FAILURE;
            }

            fs::path floatPath(path);
            fs::path headerPath = floatPath;
            headerPath.replace_extension(".hdr");

            if (!fs::exists(floatPath) || !fs::exists(headerPath))
            {
                return geoStatus::FAILURE;
            }

            // Make paths cannonical
            headerPath = fs::canonical(headerPath);
            floatPath = fs::canonical(floatPath);

            size_t headerSize = fs::file_size(headerPath);

            size_t dataSize = fs::file_size(floatPath);

            if (!headerSize || !dataSize)
            {
                return geoStatus::FAILURE;
            }

            FILE *fp = fopen(headerPath.string().c_str(), "rb");
            if (fp == nullptr)
            {
                cerr << "Unable top open header " << path << endl;
                return geoStatus::FAILURE;
            }

            // Load header from file
            BinaryHeader h(fp, headerSize);

            // Check if header is valid
            if (!h.valid())
            {
                cerr << path << " does not contain a valid header." << endl;
                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Header is valid

            // Get parameters from the header
            auto [rows, columns, x0, y0, dxDeg, dyDeg, noData] = h.getParameters();

            // Get how many meters has 1 arcsec at y0 latitude
            auto [lonMeters, latMeters] = arcSecMeters(y0);

            // Calculate dx in meters: (round up)
            auto dx = ceilf(
                dxDeg * 3600 // Convert dxDeg to arc secs
                * lonMeters  // Multiply by how many lon meters are there in 1 arcsec at this lat
            );

            // Calculate dy in meters: (round up)
            auto dy = ceilf(
                dyDeg * 3600 // Convert dyDeg to arc secs
                * latMeters  // Multiply by how many lat meters are there in 1 arcsec at this lat
            );

            // Float data
            auto [status, sz, data] = DataSet<float>::loadBinary(floatPath.string());

            // Check if the whole file was loaded
            // If not, discard partial loaded grid
            if (sz < rows * columns)
            {
                // ifDebug([&]
                //         { cerr
                //               << "Warning! "
                //               << floatPath.stem().string()
                //               << " should contain "
                //               << rows * columns
                //               << " values, but only "
                //               << sz << " values were read.\nReleasing partial data..." << endl; });

                // Dispose partially read data
                free(data);
                fclose(fp);
                return geoStatus::FAILURE;
            }

            Grid::setup(GridFormat::ESRI_FLOAT, grid, data, rows, columns, x0, y0, dx, dy, dxDeg, dyDeg, noData);

            // ESRI binary stores last row at the top, rows need to be reversed.
            grid.reverseRows();

            // Close file pointer
            fclose(fp);
            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a 2D grid into an ESRI ASCII file
         * @param path to save the file
         * @param data 2D flattened array (rows * columns), row-major
         * @param rows Grid rows (cells in Y direction)
         * @param columns Grid columns (cells in X direction
         * @param x0 X coordinate (longitude) of the lower left corner of the grid
         * @param y0 Y coordinate (latitude) of the lower left corner of the grid
         * @param dxDeg X resolution (decimal degrees)
         * @param dyDeg Y resolution (decimal degrees)
         * @param nodata value to be considered as NODATA
         */
        static geoStatus saveAscii(
            const char *path,
            const float *data,
            int rows,
            int columns,
            double x0,
            double y0,
            double dxDeg,
            double dyDeg,
            float nodata = NAN)
        {

            size_t count = rows * columns;
            if (data == nullptr || count == 0)
            {
                // ifDebug([&]
                //         { cerr << "Grid is empty, nothing to save" << endl; });

                return geoStatus::FAILURE;
            }

            fs::path dataP(path);

            // Override extension!
            dataP.replace_extension(".asc");

            // Use low level primitives to improve performance

            FILE *fp = fopen(dataP.string().c_str(), "w");

            if (fp == NULL)
            {
                cerr << "Unable to open file " << dataP.string().c_str() << endl;
                return geoStatus::FAILURE;
            }

            // Write ASCII grid header
            fprintf(fp, "ncols %d\n", columns);
            fprintf(fp, "nrows %d\n", rows);
            fprintf(fp, "xllcorner %.16lf\n", x0);
            fprintf(fp, "yllcorner %.16lf\n", y0);

            // Check if cells are squared
            float percentDiff = (fabs((dxDeg - dyDeg) / dxDeg)) * 100.0f;
            // If relative percent difference is greater than 0.1 percent, use dx, dy
            if (percentDiff > 0.1f)
            {
                // Non-squared cells
                // Use dx/dy if cell sizes are not square
                fprintf(fp, "dx %.16lf\n", dxDeg);
                fprintf(fp, "dy %.16lf\n", dyDeg);
            }
            else
            {
                // Squared cells, use cellsize
                double cellSize = std::min(dxDeg, dyDeg);
                fprintf(fp, "cellsize %.16f\n", cellSize);
            }

            fprintf(fp, "NODATA_value %7f\n", nodata);

            // Save rows in reverse order
            geoStatus status = DataSet<float>::saveTextReverseBatches(fp, (size_t)0, data, count, columns);

            // ifDebug([&]
            //         { cout << endl
            //                << "Written " << dataP.string() << endl; });

            fclose(fp);

            // Save projection file
            saveWGS84Projection(dataP.string().c_str());

            // Return success even if projection file couldn't be created.
            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a grid into an ASCII file
         * @param grid Grid
         * @param path Path to save the grid
         */
        static geoStatus saveAscii(const Grid &grid, const string &path)
        {

            auto [x0, y0, xMax, yMax] = grid.extents();
            auto [dxDeg, dyDeg] = grid.resolution();
            auto [rows, columns] = grid.dimensions();
            auto noData = grid.noDataValue();
            auto data = grid.c_float();

            return saveAscii(path.c_str(), data, rows, columns, x0, y0, dxDeg, dyDeg, noData);
        }

        /**
         * @brief Saves a 2D grid into an ESRI 32-bit float binary file (.bil, .hdr)
         * @param path Output file path
         * @param data Grid data
         * @param rows Grid rows
         * @param columns Grid columns
         * @param x0 Lower left corner longitude
         * @param y0 Lower left corner latitude
         * @param dxDeg Grid X resolution - decimal degrees
         * @param dyDeg  Grid Y resolution - decimal degrees
         * @param nodata NoData value
         * @return status status::SUCCESS if saving succeeded, status::FAILURE otherwise
         */
        static geoStatus saveFloat(
            const char *path,
            const float *data,
            int rows,
            int columns,
            double x0,
            double y0,
            double dxDeg,
            double dyDeg,
            float nodata = NAN)
        {

            int count = rows * columns;
            if (count <= 0)

                if (data == nullptr || count == 0)
                {
                    // ifDebug([&]
                    //         { cerr << "Grid is empty, nothing to save" << endl; });

                    return geoStatus::FAILURE;
                }

            // Controls flush over the output byte stream

            // Header ASCII file (.hdr extension)
            fs::path headerP(path);
            headerP.replace_extension(".hdr");

            // Raw data file (.bil extension)
            fs::path dataP = headerP;
            dataP.replace_extension(".bil");

            // Use low level primitives to improve performance

            // Write header file first

            std::FILE *fp = std::fopen(headerP.string().c_str(), "w");

            if (fp == NULL)
            {
                cerr << "Unable to open file " << headerP.string().c_str() << endl;
                return geoStatus::FAILURE;
            }

            // Ulx: center of the top left cell
            // Uly: center of the top left cell
            double ulx = x0 + (dxDeg / 2.0f);
            double uly = y0 + (float(rows - 1) * dyDeg) + (dyDeg / 2.0f);

            int rowBytes = columns * sizeof(float);

            // Write BIL header
            // see https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
            fprintf(fp, "byteorder      i\n");
            fprintf(fp, "layout         bil\n");
            fprintf(fp, "nrows          %d\n", rows);
            fprintf(fp, "ncols          %d\n", columns);
            fprintf(fp, "nbands         1\n");            // Single band
            fprintf(fp, "nbits          32\n");           // 32 bits - Float
            fprintf(fp, "bandrowbytes   %d\n", rowBytes); // Bytes of each band row: colums * sizeof(float)
            fprintf(fp, "totalrowbytes  %d\n", rowBytes); // Single band, same as band row bytes
            fprintf(fp, "pixeltype      float\n");        // Float data type
            fprintf(fp, "ulxmap         %.16lf\n", ulx);
            fprintf(fp, "ulymap         %.16lf\n", uly);
            fprintf(fp, "xdim           %.16lf\n", dxDeg);
            fprintf(fp, "ydim           %.16lf\n", dyDeg);
            fprintf(fp, "nodata         %f\n", nodata);
            fclose(fp);

            // Now write raw data file - binary mode
            fp = std::fopen(dataP.string().c_str(), "wb");

            if (fp == NULL)
            {
                // Error! remove header file
                // ifDebug([&]
                //         { cerr << "Unable to open file " << dataP.string().c_str() << endl; });

                fs::remove(headerP);
                return geoStatus::FAILURE;
            }

            // Write binary data in reverse order
            geoStatus status = DataSet<float>::saveBinaryReverse(fp, 0, data, count, columns);

            fclose(fp);

            // if (writeFailed)
            if (status != geoStatus::SUCCESS)
            {
                // Raw data writing failed, remove header file
                fs::remove(headerP);
                fs::remove(dataP);
                return geoStatus::FAILURE;
            }

            // ifDebug([&]
            //         { cout << endl
            //                << "Written " << dataP.string() << endl; });

            // Save projection file
            saveWGS84Projection(path);

            // Return success even if projection file couldn't be created.
            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a grid into float-32 file
         * @param grid Grid
         * @param path Path to save the grid
         */
        static geoStatus saveFloat(const Grid &grid, const string &path)
        {

            auto [x0, y0, xMax, yMax] = grid.extents();
            auto [dxDeg, dyDeg] = grid.resolution();
            auto [rows, columns] = grid.dimensions();
            auto noData = grid.noDataValue();
            auto data = grid.c_float();

            return saveFloat(path.c_str(), data, rows, columns, x0, y0, dxDeg, dyDeg, noData);
        }

    }; // End struct Esri

    /**
     * @brief ENVI grid
     *
     */
    struct Envi
    {

        /** @brief NAN value */
        static constexpr auto nan = NAN;

        /**
         * @brief ENVI grid data type
         *
         */
        enum class DataType
        {
            u8 = 1,   // Byte: 8-bit unsigned integer
            i16 = 2,  // Integer: 16-bit signed integer
            i32 = 3,  // Long: 32-bit signed integer
            f32 = 4,  // Floating-point: 32-bit single-precision
            d64 = 5,  // Double-precision: 64-bit double-precision floating-point
            ff32 = 6, // Complex: Real-imaginary pair of single-precision floating-point
            dd64 = 9, // Double-precision complex: Real-imaginary pair of double precision floating-point
            u16 = 12, // Unsigned integer: 16-bit
            u32 = 13, // Unsigned long integer: 32-bit
            i64 = 14, // 64-bit long integer (signed)
            u64 = 15  // 64-bit unsigned long integer (unsigned)
        };

        /**
         * @brief ENVI file byte order
         *
         */
        enum class ByteOrder
        {
            LE = 0,
            BE = 1
        };

        /**
         * @brief File header
         *
         */
        class Header : public Options
        {
        public:
            /**
             * @brief Construct a new Header object from a single or multiline text
             *
             * @param text Header text
             */
            Header(const string &text = "") : Options(text, '=')
            {
            }

            /**
             * @brief Construct a new Header object from a file pointer
             *
             * @param fp Pointer to opened header file
             * @param fileSize File size in bytes
             */
            Header(FILE *fp, size_t fileSize) : Header("")
            {
                parseFile(fp, fileSize);
            }

            /**
             * @brief Parses the header from a file pointer
             * @param fp Pointer to already opened file
             * @param fileSize Estimated file size
             */
            void parseFile(FILE *fp, size_t fileSize)
            {

                // No valid file stream, return.
                if (fp == NULL)
                {
                    return;
                }

                // Header is always at the start of the file
                fseek(fp, 0L, SEEK_SET);

                string headerString;

                auto size = Strings::loadFile(fp, fileSize, headerString);

                if (size == 0)
                {
                    return;
                }

                this->parse(headerString);
            }

            /**
             * @brief Gets the value for a key, removing enclosing braces if present.
             *
             * @param key Key
             * @return string Value for the key
             */
            string get(const string &key) override
            {
                string value = Options::get(key);

                // Remove enclosing braces
                std::regex braces("^\\{(.*)\\}$");

                if (std::regex_match(value, braces))
                {
                    value.erase(value.begin());
                    value.erase(value.end() - 1);
                }

                return value;
            }

            /**
             * @brief Parses the provided string using the provided equal sign
             *
             * @param data String to parse
             * @param equalSign Equal sign
             */
            void parse(const string &data, char equalSign = '=') override
            {
                //   ENVI header is valid only if the first line starts with "ENVI"

                auto lines = Strings::splitLines(data);

                if (!lines.size())
                {
                    return;
                }

                if (Strings::tolower(Strings::trim(lines[0])).compare("envi") != 0)
                {
                    // First line must be ENVI
                    return;
                }

                // Remove first line (ENVI)
                lines.erase(lines.begin());

                Options::parse(Strings::join(lines, "\n"), this->equalSign);

                // No key / values defined? return.
                if (this->empty())
                {
                    return;
                }

                if (this->contains("lines") && this->contains("samples"))
                {
                    this->rows = this->getInt("lines");
                    this->columns = this->getInt("samples");
                    if (this->rows > 0 && this->columns > 0)
                    {
                        this->dimensionDefined = true;
                    }
                    else
                    {
                        // rows or cols are zero
                        this->rows = 0;
                        this->columns = 0;
                    }
                }

                if (this->contains("map info"))
                {
                    vector<string> mapInfo = Strings::split(this->get("map info"), ",");
                    // MapInfo must contain 8 values, the last one must be "wgs-84" in lowercase
                    if (mapInfo.size() == 8 &&
                        Strings::tolower(Strings::trim(mapInfo[7])).compare("wgs-84") == 0)
                    {

                        // Get values as double to calculate y0
                        double x0 = std::stod(mapInfo[3]);
                        double yMax = std::stod(mapInfo[4]);
                        double dx = std::stod(mapInfo[5]);
                        double dy = std::stod(mapInfo[6]);

                        this->x0 = x0;
                        this->yMax = yMax;
                        this->dx = dx;
                        this->dy = dy;

                        this->xMax = x0 + (this->columns) * dx;
                        this->y0 = yMax - (this->rows) * dy;

                        this->originDefined = true;
                        this->resolutionDefined = true;
                    }
                }

                if (this->contains("data ignore value"))
                {
                    this->noData = this->getFloat("data ignore value");
                    this->noDataDefined = true;
                }
            }

            /**
             * @brief Checks if this object is valid
             *
             * @return true if it contains at least one key
             * @return false if it is empty
             */
            bool valid() override
            {
                if (!Options::valid())
                {
                    return false;
                }

                if (!this->dimensionDefined || !this->originDefined || !this->resolutionDefined || !this->noDataDefined)
                {
                    return false;
                }
                return true;
            }

            /**
             * @brief Returns this header parameters
             *
             * @return tuple<int, int, double, double, double, double, float, int>
             */
            tuple<int, int, double, double, double, double, float, int> getParameters()
            {
                return {rows, columns, x0, y0, dx, dy, noData, dataType};
            }

        private:
            int rows{};      /*!< Count of rows */
            int columns{};   /*!< Count of columns */
            double x0{};     /*!< left  x coordinate*/
            double xMax{};   /*!< right x coordinate */
            double y0{};     /*!< bottom y coordinate */
            double yMax{};   /*!< top y coordinate */
            double dx{};     /*!< cell size in x */
            double dy{};     /*!< cell size in y */
            float noData{};  /*!< nodata value*/
            int dataType{4}; // Assume 4 = float (5 = double)

            bool dimensionDefined{false};  /*!< True when dimension is defined */
            bool originDefined{false};     /*!< True when origin is defined */
            bool resolutionDefined{false}; /*!< True when resolution is defined */
            bool noDataDefined{false};     /*!< True when NODATA value is defined*/
        };

        /**
         * @brief Loads an ENVI 32 or 64-bit floating point grid (.ftl, .hdr)
         * @param grid Reference to the grid instance to load data into
         * @param path Path to the binary file (extension is optional)
         * @return status
         */
        static geoStatus loadBinary(Grid &grid, const string &path)
        {

            if (!path.length())
            {
                return geoStatus::FAILURE;
            }

            fs::path floatPath(path);
            fs::path headerPath = floatPath;
            headerPath.replace_extension(".hdr");

            if (!fs::exists(floatPath) || !fs::exists(headerPath))
            {
                return geoStatus::FAILURE;
            }

            // Make paths cannonical
            headerPath = fs::canonical(headerPath);
            floatPath = fs::canonical(floatPath);

            size_t headerSize = fs::file_size(headerPath);

            size_t dataSize = fs::file_size(floatPath);

            if (!headerSize || !dataSize)
            {
                return geoStatus::FAILURE;
            }

            FILE *fp = fopen(headerPath.string().c_str(), "r");
            if (fp == nullptr)
            {
                cerr << "Unable top open header " << headerPath.string() << endl;
                return geoStatus::FAILURE;
            }

            // Load header from file
            Header h(fp, headerSize);

            // Check if header is valid
            if (!h.valid())
            {
                // ifDebug([&]
                //         { cerr << path << " does not contain a valid ENVI ASCII header." << endl; });

                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Get parameters from the ENVI header
            auto [rows, columns, x0, y0, dxDeg, dyDeg, noData, dataType] = h.getParameters();

            // Get how many meters has 1 arcsec at y0 latitude
            auto [lonMeters, latMeters] = arcSecMeters(y0);

            // Calculate dx in meters:
            auto dx = floorf(
                dxDeg * 3600 // Convert dxDeg to arc secs
                * lonMeters  // Multiply by how many lon meters are there in 1 arcsec at this lat
            );

            // Calculate dy in meters:
            auto dy = floorf(
                dyDeg * 3600 // Convert dyDeg to arc secs
                * latMeters  // Multiply by how many lat meters are there in 1 arcsec at this lat
            );

            // Load actual data
            if (dataType == 4)
            {
                // Float data
                auto [status, sz, data] = DataSet<float>::loadBinary(floatPath.string());

                // Check if the whole file was loaded
                // If not, discard partial loaded grid
                if (sz < rows * columns)
                {
                    // ifDebug([&]
                    //         { cerr
                    //               << "Warning! "
                    //               << floatPath.stem().string()
                    //               << " should contain "
                    //               << rows * columns
                    //               << " values, but only "
                    //               << sz << " values were read.\nReleasing partial data..." << endl; });

                    // Dispose partially read data
                    free(data);
                    fclose(fp);
                    return geoStatus::FAILURE;
                }

                // Assign data
                Grid::setup(GridFormat::ENVI_FLOAT, grid, data, rows, columns, x0, y0, dx, dy, dxDeg, dyDeg, noData);
                // ENVI stores last row at the top, rows need to be reversed.
                grid.reverseRows();

                // Close file pointer
                fclose(fp);
                return geoStatus::SUCCESS;
            }
            else if (dataType == 5)
            {
                // Double data
                auto [status, sz, doubleData] = DataSet<double>::loadBinary(floatPath.string());

                if (sz < rows * columns)
                {
                    cerr
                        << "Warning! "
                        << floatPath.stem().string()
                        << " should contain "
                        << rows * columns
                        << " values, but only "
                        << sz << " values were read.\nReleasing partial data..." << endl;
                    // Dispose partial read data
                    free(doubleData);
                    fclose(fp);
                    return geoStatus::FAILURE;
                }

                // A new float array needs to be allocated and double data needs to be copied to float array
                float *data = (float *)malloc(sz * sizeof(float));
                if (data == nullptr)
                {
                    free(doubleData);
                    return geoStatus::FAILURE;
                }
                // Copy double data to float data
                for (size_t i = 0; i < sz; i++)
                {
                    data[i] = static_cast<float>(doubleData[i]);
                }

                // Release double data
                free(doubleData);

                // Setup grid
                Grid::setup(GridFormat::ENVI_DOUBLE, grid, data, rows, columns, x0, y0, dx, dy, dxDeg, dyDeg, noData);

                // ENVI stores last row at the top, rows need to be reversed.
                grid.reverseRows();

                return geoStatus::SUCCESS;
            }

            return geoStatus::FAILURE;
        }

        /**
         * @brief Saves a 2D grid into an ENVI 32-bit float binary file (.flt, .hdr)
         * @param path Output file path
         * @param data Grid data
         * @param rows Grid rows
         * @param columns Grid columns
         * @param x0 Lower left corner longitude
         * @param y0 Lower left corner latitude
         * @param dxDeg Grid X resolution - decimal degrees
         * @param dyDeg  Grid Y resolution - decimal degrees
         * @param nodata NoData value
         * @return status status::SUCCESS if saving succeeded, status::FAILURE otherwise
         */
        static geoStatus saveFloat(
            const char *path,
            const float *data,
            int rows,
            int columns,
            double x0,
            double y0,
            double dxDeg,
            double dyDeg,
            float nodata = NAN)
        {

            int count = rows * columns;

            if (data == nullptr || count == 0)
            {
                // ifDebug([&]
                //         { cerr << "Grid is empty, nothing to save" << endl; });

                return geoStatus::FAILURE;
            }

            // Header ASCII file (.hdr extension)
            fs::path headerP(path);
            headerP.replace_extension(".hdr");

            // Raw data file (.flt extension)
            fs::path dataP = headerP;
            dataP.replace_extension(".flt");

            // Use low level primitives to improve performance

            // Write header file first

            std::FILE *fp = std::fopen(headerP.string().c_str(), "w");

            if (fp == NULL)
            {
                cerr << "Unable to open file " << headerP.string().c_str() << endl;
                return geoStatus::FAILURE;
            }

            // Top of the last cell
            double dyMax = y0 + ((double)rows * dyDeg);

            // Write ENVI header
            // see https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html
            fprintf(fp, "ENVI\n");
            fprintf(fp, "samples = %d\n", columns);
            fprintf(fp, "lines   = %d\n", rows);
            fprintf(fp, "bands   = 1\n"); // Single band
            fprintf(fp, "header offset = 0\n");
            fprintf(fp, "file type = ENVI Standard\n");
            fprintf(fp, "data type = 4\n");    // 4 = Single-precision: 32-bit single-precision floating-point
            fprintf(fp, "interleave = bil\n"); // Bil, on single band is the same as bsq
            fprintf(fp, "byte order = 0\n");   // little endian
            fprintf(fp,
                    "map info = {Geographic Lat/Lon, 1, 1, %.16lf, %.16lf, %.16lf, %.16lf,WGS-84}\n",
                    x0,
                    dyMax, dxDeg, dyDeg);
            fprintf(fp, "coordinate system string = {\n"
                        "GEOGCS[\"GCS_WGS_1984\",\n"
                        "DATUM[\"D_WGS_1984\",\n"
                        "SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],\n"
                        "PRIMEM[\"Greenwich\",0.0],\n"
                        "UNIT[\"Degree\",0.0174532925199433]]}\n");
            fprintf(fp, "data ignore value = %f\n", nodata);
            fprintf(fp, "x start %.16lf", x0);
            fprintf(fp, "y start %.16lf", dyMax);
            fclose(fp);

            // Now write raw data file - Binary mode
            fp = std::fopen(dataP.string().c_str(), "wb");

            if (fp == NULL)
            {
                // Error! remove header file
                cerr << "Unable to open file " << dataP.string().c_str() << endl;
                fs::remove(headerP);
                return geoStatus::FAILURE;
            }

            // Write binary data in reverse order
            geoStatus status = DataSet<float>::saveBinaryReverse(fp, 0, data, count, columns);

            fclose(fp);

            // if (writeFailed)
            if (status != geoStatus::SUCCESS)
            {
                // Raw data writing failed, remove header file
                fs::remove(headerP);
                fs::remove(dataP);
                return geoStatus::FAILURE;
            }

            // ifDebug([&]
            //         { cout << "Written " << dataP.string() << endl; });

            // Save projection file
            saveWGS84Projection(path);

            // Return success even if projection file couldn't be created.
            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a 2D grid into an ENVI 64-bit double binary file (.flt, .hdr)
         * @param path Output file path
         * @param data Grid data
         * @param rows Grid rows
         * @param columns Grid columns
         * @param x0 Lower left corner longitude
         * @param y0 Lower left corner latitude
         * @param dxDeg Grid X resolution - decimal degrees
         * @param dyDeg  Grid Y resolution - decimal degrees
         * @param nodata NoData value
         * @return status status::SUCCESS if saving succeeded, status::FAILURE otherwise
         */
        static geoStatus saveDouble(
            const char *path,
            const float *data,
            int rows,
            int columns,
            double x0,
            double y0,
            double dxDeg,
            double dyDeg,
            float nodata = NAN)
        {

            int count = rows * columns;

            if (data == nullptr || count == 0)
            {
                // ifDebug([&]
                //         { cerr << "Grid is empty, nothing to save" << endl; });

                return geoStatus::FAILURE;
            }

            // Header ASCII file (.hdr extension)
            fs::path headerP(path);
            headerP.replace_extension(".hdr");

            // Raw data file (.flt extension)
            fs::path dataP = headerP;
            dataP.replace_extension(".flt");

            // Use low level primitives to improve performance

            // Write header file first

            std::FILE *fp = std::fopen(headerP.string().c_str(), "w");

            if (fp == NULL)
            {
                cerr << "Unable to open file " << headerP.string().c_str() << endl;
                return geoStatus::FAILURE;
            }

            // Top of the last cell
            double dyMax = y0 + ((double)rows * dyDeg);

            // Write ENVI header
            // see https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html
            fprintf(fp, "ENVI\n");
            fprintf(fp, "samples = %d\n", columns);
            fprintf(fp, "lines   = %d\n", rows);
            fprintf(fp, "bands   = 1\n"); // Single band
            fprintf(fp, "header offset = 0\n");
            fprintf(fp, "file type = ENVI Standard\n");
            fprintf(fp, "data type = 5\n");    // 5 = Double-precision: 64-bit single-precision floating-point
            fprintf(fp, "interleave = bil\n"); // Bil, on single band is the same as bsq
            fprintf(fp, "byte order = 0\n");   // little endian
            fprintf(fp,
                    "map info = {Geographic Lat/Lon, 1, 1, %.16lf, %.16lf, %.16lf, %.16lf,WGS-84}\n",
                    x0,
                    dyMax, dxDeg, dyDeg);
            fprintf(fp, "coordinate system string = {\n"
                        "GEOGCS[\"GCS_WGS_1984\",\n"
                        "DATUM[\"D_WGS_1984\",\n"
                        "SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],\n"
                        "PRIMEM[\"Greenwich\",0.0],\n"
                        "UNIT[\"Degree\",0.0174532925199433]]}\n");
            fprintf(fp, "data ignore value = %f\n", nodata);
            fprintf(fp, "x start %.16lf", x0);
            fprintf(fp, "y start %.16lf", dyMax);
            fclose(fp);

            // Now write raw data file - Binary mode
            fp = std::fopen(dataP.string().c_str(), "wb");

            if (fp == NULL)
            {
                // Error! remove header file
                cerr << "Unable to open file " << dataP.string().c_str() << endl;
                fs::remove(headerP);
                return geoStatus::FAILURE;
            }

            // Write binary data in reverse order
            geoStatus status = DataSet<float>::saveBinaryReverse(fp, 0, data, count, columns);

            fclose(fp);

            // if (writeFailed)
            if (status != geoStatus::SUCCESS)
            {
                // Raw data writing failed, remove header file
                fs::remove(headerP);
                fs::remove(dataP);
                return geoStatus::FAILURE;
            }

            // ifDebug([&]
            //         { cout << "Written " << dataP.string() << endl; });

            // Save projection file
            saveWGS84Projection(path);

            // Return success even if projection file couldn't be created.
            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a grid into float-32 file
         * @param grid Grid
         * @param path Path to save the grid
         */
        static geoStatus saveFloat(const Grid &grid, const string &path)
        {

            auto [x0, y0, xMax, yMax] = grid.extents();
            auto [dxDeg, dyDeg] = grid.resolution();
            auto [rows, columns] = grid.dimensions();
            auto noData = grid.noDataValue();
            auto data = grid.c_float();

            return saveFloat(path.c_str(), data, rows, columns, x0, y0, dxDeg, dyDeg, noData);
        }

        /**
         * @brief Saves a grid into float-32 file
         * @param grid Grid
         * @param path Path to save the grid
         */
        static geoStatus saveDouble(const Grid &grid, const string &path)
        {

            auto [x0, y0, xMax, yMax] = grid.extents();
            auto [dxDeg, dyDeg] = grid.resolution();
            auto [rows, columns] = grid.dimensions();
            auto noData = grid.noDataValue();
            auto data = grid.c_float();

            return saveDouble(path.c_str(), data, rows, columns, x0, y0, dxDeg, dyDeg, noData);
        }

    }; // End struct Envi

    /**
     * @brief Surfer grids
     *
     */
    struct Surfer
    {

        /** @brief NAN value */
        static constexpr auto nan = 1.70141e+038f;

        /** @brief Value to be considered as zero  */
        static constexpr auto epsilon = 1e-8f;

        /** @brief Surfer file type */
        enum class fileType
        {
            FLOAT,  /*!< Surfer 6 binary (float z values) */
            DOUBLE, /*!< Surfer 7 binary (double z values)*/
            TEXT    /*!< Surfer 6 ASCII */
        };

        /**
         * @brief Surfer ASCII header
         * @see https://grapherhelp.goldensoftware.com/subsys/ascii_grid_file_format.htm
         * Undocumented: xmin, xmax, ymin, ymax are at the center of the cell.
         */
        class Header : public Options
        {
        public:
            /**
             * @brief Construct a new Header instance
             *
             * @param text Header text
             */
            Header(const string &text = "") : Options(text, ' '), type(fileType::TEXT)
            {
            }

            /**
             * @brief Construct a new Header object from a file pointer
             *
             * @param fp Pointer to opened header file
             * @param fileSize File size in bytes
             */
            Header(FILE *fp, size_t fileSize) : Header("")
            {
                parseFile(fp, fileSize);
            }

            /**
             * @brief Parses the header from a file pointer
             *
             * @param fp Pointer to already opened file
             * @param fileSize File size
             */
            void parseFile(FILE *fp, size_t fileSize)
            {

                // No valid file stream, return.
                if (fp == NULL)
                {
                    return;
                }

                // Header is always at the start of the file
                fseek(fp, 0L, SEEK_SET);

                char magic[4];

                // Read magic number
                if (fread(magic, sizeof(char), 4, fp) != 4)
                {
                    return;
                }

                // Read magic string at the start of the file
                string magicStr(magic, 4);

                magicStr = Strings::tolower(Strings::trim(magicStr));

                if (magicStr.compare("dsaa") == 0)
                {
                    this->parseTextHeader(fp);
                }
                else if (magicStr.compare("dsbb") == 0)
                {
                    this->parseBinary32Header(fp);
                }
                else if (magicStr.compare("dsrb") == 0)
                {
                    this->parseBinary64Header(fp);
                }
            }

            /**
             * @brief Parse 32-bit binary or text header data
             * @param data String data (ignored)
             * @param equalSign (ignored)
             */
            void parse(const string &data, char equalSign = ' ') override
            {

                // x0Center, xMaxCenter, y0Center and yMaxCenter attributes
                // already set from parseTextHeader or parseBinary32Header

                double xMinCenter = this->x0Center;
                double xMaxCenter = this->xMaxCenter;
                double yMinCenter = this->y0Center;
                double yMaxCenter = this->yMaxCenter;

                this->dx = (xMaxCenter - xMinCenter) / ((double)(columns - 1));
                this->dy = (yMaxCenter - yMinCenter) / ((double)(rows - 1));

                this->x0 = this->x0Center - (this->dx / 2.0f);
                this->y0 = this->y0Center - (this->dy / 2.0f);

                this->xMax = this->xMaxCenter + (this->dx / 2.0f);
                this->yMax = this->yMaxCenter + (this->dy / 2.0f);

                this->noData = 1.70141e+38f; // Surfer blank value

                this->dimensionDefined =
                    this->originDefined =
                        this->resolutionDefined =
                            this->noDataDefined = true;
            }

            /**
             * @brief Checks if this object is valid
             *
             * @return true if it contains at least one key
             * @return false if it is empty
             */
            bool valid() override
            {
                if (!this->dimensionDefined || !this->originDefined || !this->resolutionDefined || !this->noDataDefined)
                {
                    return false;
                }
                return true;
            }

            /**
             * @brief Get the Parameters object
             *
             * @return tuple<int, int, double, double, double, double, float, fileType> rows, columns, x0, y0, dx, dy, nodata, file type
             */
            tuple<int, int, double, double, double, double, float, fileType> getParameters()
            {
                return {rows, columns, x0, y0, dx, dy, noData, type};
            }

        private:
            /**
             * @brief Parse text header (Surfer 6)
             * @param fp File pointer positioned right after header string DSAA
             */
            void parseTextHeader(FILE *fp)
            {
                string headerString;

                if (fscanf(fp, "%d%d%lf%lf%lf%lf%lf%lf",
                           &this->columns, &this->rows,
                           &this->x0Center, &this->xMaxCenter,
                           &this->y0Center, &this->yMaxCenter,
                           &this->z0, &this->zMax) != 8)
                {
                    return;
                }

                if (this->rows == 0 || this->columns == 0)
                {
                    return;
                }

                // Set dataType to binary
                this->type = fileType::TEXT;

                this->parse(headerString);
            }

            /**
             * @brief Parse 32-bit binary header (Surfer 6)
             * @param fp File pointer positioned right after binary header string DSBB
             */
            void parseBinary32Header(FILE *fp)
            {
                string headerString;

                if (fread(&this->columns, sizeof(short), 1, fp) != 1)
                {
                    return;
                }

                if (fread(&this->rows, sizeof(short), 1, fp) != 1)
                {
                    return;
                }

                if (fread(&this->x0Center, sizeof(double), 1, fp) != 1)
                {
                    return;
                }

                if (fread(&this->xMaxCenter, sizeof(double), 1, fp) != 1)
                {
                    return;
                }

                if (fread(&this->y0Center, sizeof(double), 1, fp) != 1)
                {
                    return;
                }

                if (fread(&this->yMaxCenter, sizeof(double), 1, fp) != 1)
                {
                    return;
                }

                if (fread(&this->z0, sizeof(double), 1, fp) != 1)
                {
                    return;
                }

                if (fread(&this->zMax, sizeof(double), 1, fp) != 1)
                {
                    return;
                }

                // Set dataType to binary
                this->type = fileType::FLOAT;

                this->parse(headerString);
            }

            /**
             * @brief Parse Surfer 7 binary header
             * @param fp Pointer to the file. Reading position must be right after header string 0x42525344 (DSRB)
             * @see https://surferhelp.goldensoftware.com/topics/surfer_7_grid_file_format.htm?tocpath=File%20Types%7CFile%20Formats%7C_____45
             */
            void parseBinary64Header(FILE *fp)
            {

                uint32_t headerSection[2];

                // Read header section
                if (fread(headerSection, sizeof(uint32_t), 2, fp) != 2)
                {
                    return;
                }

                // Format of grid section:
                // Header (4 bytes) 0x44495247 string GRID
                // size    long (4 bytes - uint32_t)
                // nrow    long
                // ncol    long
                // xll     double
                // yll     double
                // xSize   double
                // ySize   double
                // zMin    double
                // zMax    double
                // rot     double
                // nodata  double  1.70141e+38
                // Total: 80 bytes
                char gridSection[80];

                if (fread(gridSection, sizeof(char), 80, fp) != 80)
                {
                    return;
                }

                // Read grid section header string (GRID) and transform to lowercase
                string gridString(gridSection, 4);
                gridString = Strings::tolower(Strings::trim(gridString));
                // Check if grid string is valid
                if (gridString.compare("grid") != 0)
                {
                    return;
                }

                // uint32_t length{};
                uint32_t nRow{};
                uint32_t nCol{};
                double xll{};
                double yll{};
                double xSize{};
                double ySize{};
                double zMin{};
                double zMax{};
                // double rotation{};
                double blankValue{nan};

                // Data is already in memory, cast the values
                size_t pos = 4;
                // Read long values
                // length = *reinterpret_cast<uint32_t *>(&gridSection[pos]);
                // skip length
                pos += sizeof(uint32_t);
                nRow = *reinterpret_cast<uint32_t *>(&gridSection[pos]);
                pos += sizeof(uint32_t);
                nCol = *reinterpret_cast<uint32_t *>(&gridSection[pos]);
                pos += sizeof(uint32_t);

                // Read double values
                xll = *reinterpret_cast<double *>(&gridSection[pos]);
                pos += sizeof(double);
                yll = *reinterpret_cast<double *>(&gridSection[pos]);
                pos += sizeof(double);
                xSize = *reinterpret_cast<double *>(&gridSection[pos]);
                pos += sizeof(double);
                ySize = *reinterpret_cast<double *>(&gridSection[pos]);
                pos += sizeof(double);
                zMin = *reinterpret_cast<double *>(&gridSection[pos]);
                pos += sizeof(double);
                zMax = *reinterpret_cast<double *>(&gridSection[pos]);
                pos += sizeof(double);
                // rotation = *reinterpret_cast<double *>(&gridSection[pos]);
                //  skip rotation
                pos += sizeof(double);
                blankValue = *reinterpret_cast<double *>(&gridSection[pos]);

                // No dimensions??
                if (nRow * nCol == 0)
                {
                    return;
                }

                // Now read data section
                char gridDataSection[4];
                uint32_t gridDataLength;

                if (fread(gridDataSection, sizeof(char), 4, fp) != 4)
                {
                    return;
                }

                string gridDataStr(gridDataSection, 4);

                gridDataStr = Strings::tolower(Strings::trim(gridDataStr));
                // Check if grid data string is valid
                if (gridDataStr.compare("data") != 0)
                {
                    return;
                }

                // Read grid data length in bytes
                if (fread(&gridDataLength, sizeof(uint32_t), 1, fp) != 1)
                {
                    return;
                }

                // Check if there is enough data to read
                if (gridDataLength != (nRow * nCol) * sizeof(double))
                {
                    return;
                }

                // Complete header definition

                this->rows = nRow;
                this->columns = nCol;
                this->dx = xSize;
                this->dy = ySize;
                this->x0Center = xll;
                this->y0Center = yll;
                this->z0 = zMin;
                this->zMax = zMax;
                this->noData = blankValue;

                // Define additional parameters
                this->x0 = this->x0Center - (this->dx / 2.0f);
                this->xMaxCenter = this->x0Center + (this->columns - 1) * this->dx;
                this->y0 = this->y0Center - (this->dy / 2.0f);
                this->yMaxCenter = this->y0Center + (this->rows - 1) * this->dy;

                this->xMax = this->x0 + (this->columns) * this->dx;
                this->yMax = this->y0 + (this->rows) * this->dy;

                // Set dataType to binary
                this->type = fileType::DOUBLE;

                // Nothing else to do, header already defined.
                this->dimensionDefined =
                    this->originDefined =
                        this->resolutionDefined =
                            this->noDataDefined = true;
            }

            int rows{};                     /*!< Grid rows */
            int columns{};                  /*!< Grid columns */
            double x0{};                    /*!< X coordinate (longitude) of the lower left corner of the grid */
            double x0Center{};              /*!< Coordinate of the center of the first x cell */
            double xMax{};                  /*!< Coordinate of the lower right corner of the grid  */
            double xMaxCenter{};            /*!< Coordinate of the center of the last x cell */
            double y0{};                    /*!< Y coordinate (latitude) of the lower left corner of the grid */
            double y0Center{};              /*!< Coordinate of the center of the first y cell */
            double yMax{};                  /*!< Coordinate of the upper y of the grid */
            double yMaxCenter{};            /*!< Coordinate of the center of the cells on the upper row*/
            double z0{};                    /*!< Minimum value on the grid */
            double zMax{};                  /*!< Maximum value on the grid */
            double dx{};                    /*!< X resolution of the grid*/
            double dy{};                    /*!< Y resolution of the grid */
            float noData{};                 /*!< NODATA value */
            fileType type{fileType::FLOAT}; /*!< Grid file type */

            bool dimensionDefined{false};  /*!< True if dimensions are defined */
            bool originDefined{false};     /*!< True if origin is defined */
            bool resolutionDefined{false}; /*!< True if resolution is defined */
            bool noDataDefined{false};     /*!< NODATA value defined */
        };

        /**
         * @brief Loads a Sufer 6 grid
         * @param grid Target grid
         * @param path Path to the Surfer 6/7 (.grd) file (ascii or binary)
         * @return status status::SUCCESS if load was successful, status::FAILURE if load fails
         */
        static geoStatus load(Grid &grid, const string &path)
        {

            fs::path p(path);

            p.replace_extension(".grd"); // Extension must be .grd

            // Check if file exists
            if (!fs::exists(p) || !fs::is_regular_file(p))
            {
                cerr << "Surfer grid " << p.string() << " not found." << endl;
                return geoStatus::FAILURE;
            }

            // Convert path to canonical
            p = fs::canonical(p);
            size_t fileSize = fs::file_size(p);

            // Open file in binary mode to scan header
            FILE *fp = fopen(p.string().c_str(), "rb");
            if (fp == nullptr)
            {
                cerr << "Unable top open " << path << endl;
                return geoStatus::FAILURE;
            }

            // Load Surfer header from file
            Header h(fp, fileSize);

            // Check if header is valid
            if (!h.valid())
            {
                cerr << path << " does not contain a valid header." << endl;
                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Header is valid

            // Get parameters from the header
            auto [rows, columns, x0, y0, dxDeg, dyDeg, noData, gridType] = h.getParameters();

            // Get how many meters has 1 arcsec at y0 latitude
            auto [lonMeters, latMeters] = arcSecMeters(y0);

            // Calculate dx in meters. Approximate to the closest greater integer
            auto dx = ceilf(
                dxDeg * 3600 // Convert dxDeg to arc secs
                * lonMeters  // Multiply by how many lon meters are there in 1 arcsec at this lat
            );

            // Calculate dy in meters. Approximate to the closest greater integer
            auto dy = ceilf(
                dyDeg * 3600 // Convert dyDeg to arc secs
                * latMeters  // Multiply by how many lat meters are there in 1 arcsec at this lat
            );

            // fp points to the data right after the header.
            // fp was opened in binary mode to allow weird positioning after reading header on Windows
            // Load rows into the data buffer in place

            size_t count;
            float *gridData = nullptr;

            GridFormat format;

            if (gridType == fileType::TEXT)
            {
                // Load text rows
                auto [status, cnt, data] = DataSet<float>::loadText(fp, fileSize);
                count = cnt;
                gridData = data;
                format = GridFormat::TEXT;
            }
            else if (gridType == fileType::FLOAT)
            {
                // Read float data
                auto [status, cnt, data] = DataSet<float>::loadBinary(fp, fileSize);
                count = cnt;
                gridData = data;
                format = GridFormat::SURFER_FLOAT;
            }
            else if (gridType == fileType::DOUBLE)
            {
                // Read double  data
                auto [status, cnt, doubleData] = DataSet<double>::loadBinary(fp, fileSize);
                count = cnt;
                // Reuse already allocated buffer
                // Use a float pointer and move data over this pointer
                float *ptr = reinterpret_cast<float *>(doubleData);
                for (size_t i = 0; i < count; i++)
                {
                    // Extract the double value and cast to float
                    float v = static_cast<float>(doubleData[i]);
                    // Write the double value in place, advance to the next float position
                    *ptr++ = v;
                }

                // Reallocate to fit the array
                gridData = (float *)realloc(doubleData, count * sizeof(float));
                if (gridData == nullptr)
                {
                    cerr << "Unable to read double data" << endl;
                    fclose(fp);
                    return geoStatus::FAILURE;
                }

                format = GridFormat::SURFER_DOUBLE;
            }
            else
            {
                // ifDebug([&]
                //         { cerr << "Unknown surfer grid type" << endl; });
                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Check if the whole file was loaded
            // If not, discard partial loaded grid
            if (count < rows * columns)
            {
                // ifDebug([&]
                //         { cerr
                //               << "Warning! "
                //               << p.stem().string()
                //               << " should contain "
                //               << rows * columns
                //               << " values, but only "
                //               << count << " values were read.\nReleasing partial data..." << endl; });
                // Dispose read data
                if (gridData != nullptr)
                {
                    free(gridData);
                }
                fclose(fp);
                return geoStatus::FAILURE;
            }

            // Close file pointer
            fclose(fp);

            Grid::setup(format, grid, gridData, rows, columns, x0, y0, dx, dy, dxDeg, dyDeg, noData);

            return geoStatus::SUCCESS;
        }

        /**
         * @brief  Saves a 2D grid into a Surfer 6 Grid (ASCII/binary) depending on grid type
         * @param path Output file path
         * @param data Grid data
         * @param rows Grid rows
         * @param columns Grid columns
         * @param x0 Lower left corner longitude
         * @param y0 Lower left corner latitude
         * @param dxDeg Grid X resolution - decimal degrees
         * @param dyDeg  Grid Y resolution - decimal degrees
         * @param fileType Type of output grid, default Surfer 6 float
         * @return status status::SUCCESS if saving succeeded, status::FAILURE otherwise
         */
        static geoStatus
        save(
            const char *path,
            const float *data,
            int rows,
            int columns,
            double x0,
            double y0,
            double dxDeg,
            double dyDeg,
            fileType fileType = fileType::FLOAT)
        {
            // Controls flush over the output byte stream
            int count = rows * columns;

            if (data == nullptr || count == 0)
            {
                // ifDebug([&]
                //         { cerr << "Grid is empty, nothing to save" << endl; });

                return geoStatus::FAILURE;
            }

            fs::path dataP(path);

            // Override extension!
            dataP.replace_extension(".grd");

            // Use low level primitives to improve performance

            FILE *fp;

            if (fileType == fileType::FLOAT || fileType == fileType::DOUBLE)
            {
                fp = fopen(dataP.string().c_str(), "wb");
            }
            else
            {
                fp = fopen(dataP.string().c_str(), "w");
            }

            if (fp == NULL)
            {
                cerr << "Unable to open file " << dataP.string().c_str() << endl;
                return geoStatus::FAILURE;
            }

            // calculate zMin, zMax
            float zMax = data[0];
            float zMin = data[0];

            size_t total = rows * columns;
            for (size_t i = 1; i < (total / 2); i++)
            {
                // Compare current position to zmax
                if (data[i] < nan && data[i] > zMax)
                {
                    zMax = data[i];
                }

                if (data[total - i] < nan && data[total - i] > zMax)
                {
                    zMax = data[total - i];
                }

                float val = fabs(data[i]);
                if (val > epsilon && val < zMin)
                {
                    zMin = data[i];
                }

                val = fabs(data[total - i]);
                if (val > epsilon && val < zMin)
                {
                    zMin = data[total - i];
                }
            }

            // Convert zMin and zMax to double
            double zdMin{};
            double zdMax{};

            if (fabs(zMin) > epsilon)
            {
                zdMin = zMin;
            }

            if (zMax < nan)
            {
                zdMax = zMax;
            }

            // (xMin, yMin) Center of the bottom left cell
            double xMin = x0 + (dxDeg / 2.0f);
            double yMin = y0 + (dyDeg / 2.0f);

            // (xmax, yMax) Center of the top right cell
            double xMax = x0 + ((columns - 1) * dxDeg) + (dxDeg / 2.0f);
            double yMax = y0 + ((rows - 1) * dyDeg) + (dyDeg / 2.0f);

            double noData = nan;

            if (fileType == fileType::TEXT)
            {
                // Write ASCII grid header
                fprintf(fp, "DSAA\n");
                fprintf(fp, "%d %d\n", columns, rows);
                fprintf(fp, "%.16lf %.16lf\n", xMin, xMax);
                fprintf(fp, "%.16lf %.16lf\n", yMin, yMax);
                fprintf(fp, "%.16lf %.16lf\n", zdMin, zdMax);
            }
            else if (fileType == fileType::DOUBLE)
            {
                uint32_t nRow = rows;
                uint32_t nCol = columns;
                uint32_t val;
                double dVal{};

                // Write Surfer 7 binary header
                fwrite("DSRB", sizeof(char), 4, fp); // Id for Header section
                val = 4;

                fwrite(&val, sizeof(uint32_t), 1, fp); // Size of header section

                val = 1;
                fwrite(&val, sizeof(uint32_t), 1, fp); // Version

                fwrite("GRID", sizeof(char), 4, fp); // ID indicating a grid section
                val = 72;

                fwrite(&val, sizeof(uint32_t), 1, fp); // Length in bytes of the grid section

                fwrite(&nRow, sizeof(uint32_t), 1, fp); // Grid section: Row
                fwrite(&nCol, sizeof(uint32_t), 1, fp); // Grid section: Row

                fwrite(&xMin, sizeof(double), 1, fp); // xll (center of the lower left grid)
                fwrite(&yMin, sizeof(double), 1, fp); // yll (center of the lower left grid)

                fwrite(&dxDeg, sizeof(double), 1, fp); // xSize
                fwrite(&dyDeg, sizeof(double), 1, fp); // ySize

                fwrite(&zdMin, sizeof(double), 1, fp); // zMMin
                fwrite(&zdMax, sizeof(double), 1, fp); // zMax

                fwrite(&dVal, sizeof(double), 1, fp); // Rotation

                fwrite(&noData, sizeof(double), 1, fp); // Blank value

                fwrite("DATA", sizeof(char), 4, fp); // ID indicating a data section
                val = (rows * columns) * sizeof(double);
                fwrite(&val, sizeof(uint32_t), 1, fp); // Length in bytes of the data section
            }
            else
            {
                // Default to float
                unsigned short sColumns = columns;
                unsigned short sRows = rows;

                // Write Surfer 6 binary header
                fwrite("DSBB", sizeof(char), 4, fp);
                fwrite(&sColumns, sizeof(unsigned short), 1, fp);
                fwrite(&sRows, sizeof(unsigned short), 1, fp);
                fwrite(&xMin, sizeof(double), 1, fp);
                fwrite(&xMax, sizeof(double), 1, fp);
                fwrite(&yMin, sizeof(double), 1, fp);
                fwrite(&yMax, sizeof(double), 1, fp);
                fwrite(&zdMin, sizeof(double), 1, fp);
                fwrite(&zdMax, sizeof(double), 1, fp);
            }

            // Save actual data

            geoStatus status;

            if (fileType == fileType::TEXT)
            {
                // Save rows as text
                status = DataSet<float>::saveText(fp, 0, data, count, columns);
            }
            else if (fileType == fileType::DOUBLE)
            {

                // Create a buffer of double values
                double *doubleData = (double *)malloc(columns * sizeof(double));
                for (int i = 0; i < rows; i++)
                {
                    // Save one float row into the double array
                    for (int j = 0; j < columns; j++)
                    {
                        int pos = (i * columns) + j;
                        // Get value as double
                        double v = data[pos];
                        doubleData[j] = v;
                    }

                    // Write the double array
                    status = DataSet<double>::saveBinary(fp, 0, doubleData, columns, columns);
                    if (status != geoStatus::SUCCESS)
                    {
                        break;
                    }
                }

                // Release double array
                free(doubleData);
            }
            else
            {
                // Save binary data
                status = DataSet<float>::saveBinary(fp, 0, data, count, columns);
            }

            fclose(fp);

            if (status != geoStatus::SUCCESS)
            {
                fs::remove(dataP);
                return geoStatus::FAILURE;
            }

            // Save projection file
            saveWGS84Projection(dataP.string().c_str());

            // Return success even if projection file couldn't be created.
            return geoStatus::SUCCESS;
        }

        /**
         * @brief Saves a grid into float-32 file
         * @param grid Grid
         * @param path Path to save the grid
         * @param fileType Output grid file type
         * @return operation status
         */
        static geoStatus save(const Grid &grid, const string &path, fileType fileType = fileType::FLOAT)
        {

            auto [x0, y0, xMax, yMax] = grid.extents();
            auto [dxDeg, dyDeg] = grid.resolution();
            auto [rows, columns] = grid.dimensions();
            auto noData = grid.noDataValue();
            auto data = grid.c_float();

            return save(path.c_str(), data, rows, columns, x0, y0, dxDeg, dyDeg, fileType);
        }

    }; // End class Surfer

    /**
     * @brief Utilities
     *
     */
    struct Util
    {

        /**
         * @brief Create a grid filled with a value
         * @param format Grid format
         * @param value Value to fill the grid
         * @param rows Grid rows
         * @param columns Grid columns
         * @param x0 Grid X (longitude) origin
         * @param y0 Grid Y (latitude) origin
         * @param dx Grid X resolution in meters
         * @param dy Grid Y resolution in meters
         * @return Grid
         */
        static inline Grid createGrid(GridFormat format, float value, int rows, int columns, double x0, double y0, double dx, double dy)
        {
            int n = rows * columns;

            float *data = (float *)malloc(n * sizeof(float));

            if (data == nullptr)
            {
                return geo::Grid();
            }

            // Fill the grid with the supplied value
            for (int i = 0; i < n; i++)
            {
                data[i] = value;
            }

            return Grid(format, data, rows, columns, x0, y0, dx, dy, NAN);
        }

        /**
         * @brief Create a Sequential Grid object. grid[i] = 0, grid[1] = 1 and so on
         * @param format Grid format
         * @param rows Grid rows
         * @param columns Grid columns
         * @param x0 Grid X (longitude) origin
         * @param y0 Grid Y (latitude) origin
         * @param dx Grid X resolution in meters
         * @param dy Grid Y resolution in meters
         * @return Grid
         */
        static inline Grid createSequentialGrid(GridFormat format, int rows, int columns, double x0, double y0, double dx, double dy)
        {
            int n = rows * columns;

            float *data = (float *)malloc(n * sizeof(float));

            if (data == nullptr)
            {
                return geo::Grid();
            }

            // Create a sequential data grid
            // at 0,0 = 0
            // at 0, columns - 1 = columns -1
            // at rows - 1, columns = (rows - 1) * columns
            // at rows -1, columns 1 = n - 1

            for (int i = 0; i < n; i++)
            {
                data[i] = i;
            }

            return Grid(format, data, rows, columns, x0, y0, dx, dy, NAN);
        }

        /**
         * @brief Checks if the data is sequential
         * @param data Data array
         * @param n Count of array elements
         * @return true when the array is sequential
         * @return false when the array data is not sequential
         */
        static bool isSequentialData(const float *data, int n)
        {

            int i;
            for (i = 0; i < n && data[i] == (float)i; i++)
                ;

            return (i == n);

            return true;
        }
    };

    /**
     * @brief Loads a grid, guessing the format from the extension.
     * @param grid Target grid
     * @param path File path
     * @return status status::SUCCESS if grid was loaded, status::FAILURE if loading failed.
     */
    static inline geoStatus LoadGrid(Grid &grid, const string &path)
    {

        fs::path filePath(path);

        string fileExt = filePath.extension().string();

        string ext = Strings::tolower(fileExt);

        if (ext.compare(".asc") == 0)
        {
            return Esri::loadAscii(grid, path);
        }
        else if (ext.compare(".bil") == 0)
        {
            return Esri::loadFloat(grid, path);
        }
        else if (ext.compare(".flt") == 0)
        {
            return Envi::loadBinary(grid, path);
        }
        else if (ext.compare(".grd") == 0)
        {
            return Surfer::load(grid, path);
        }
        return geoStatus::FAILURE;
    }

    /**
     * @brief Loads a grid
     * @param grid Target grid
     * @param path File path
     * @param format Grid format
     * @return status status::SUCCESS if grid was loaded, status::FAILURE if loading failed.
     */
    static inline geoStatus LoadGrid(Grid &grid, const string &path, const GridFormat format)
    {
        if (format == GridFormat::ESRI_ASCII)
        {
            return Esri::loadAscii(grid, path);
        }
        else if (format == GridFormat::ESRI_FLOAT)
        {
            return Esri::loadFloat(grid, path);
        }
        else if (format == GridFormat::ENVI_FLOAT || format == GridFormat::ENVI_DOUBLE)
        {
            return Envi::loadBinary(grid, path);
        }
        else if (format == GridFormat::SURFER_ASCII || format == GridFormat::SURFER_FLOAT || format == GridFormat::SURFER_DOUBLE)
        {
            return Surfer::load(grid, path);
        }
        return geoStatus::FAILURE;
    }

    /**
     * @brief Saves the grid
     *
     * @param grid Grid to save
     * @param path Path of the output file
     * @param format Grid format
     * @return status status::SUCCESS if saving succeeds, status::FAILURE otherwise
     */
    static inline geoStatus SaveGrid(const Grid &grid, const string &path, const GridFormat format)
    {
        if (format == GridFormat::ESRI_ASCII)
        {
            return Esri::saveAscii(grid, path);
        }
        else if (format == GridFormat::ESRI_FLOAT)
        {
            return Esri::saveFloat(grid, path);
        }
        else if (format == GridFormat::ENVI_FLOAT)
        {
            return Envi::saveFloat(grid, path);
        }
        else if (format == GridFormat::ENVI_DOUBLE)
        {
            return Envi::saveDouble(grid, path);
        }
        else if (format == GridFormat::SURFER_ASCII || format == GridFormat::SURFER_FLOAT || format == GridFormat::SURFER_DOUBLE)
        {
            Surfer::fileType fileType = Surfer::fileType::FLOAT;
            if (format == GridFormat::SURFER_ASCII)
            {
                fileType = Surfer::fileType::TEXT;
            }
            else if (format == GridFormat::SURFER_DOUBLE)
            {
                fileType = Surfer::fileType::DOUBLE;
            }
            return Surfer::save(grid, path, fileType);
        }
        else if (format == GridFormat::TEXT || format == GridFormat::TEXT_REVERSE)
        {
            if (format == GridFormat::TEXT)
            {
                return grid.saveText(path.c_str());
            }
            else
            {
                return grid.saveText(path.c_str(), true);
            }
        }
        return geoStatus::FAILURE;
    }
}

#endif