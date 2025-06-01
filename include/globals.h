/**
 * @file
 * @brief Global definitions for this context
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 */
#ifndef GLOBALS_H
#define GLOBALS_H

#include <cmath>
#include <tuple>

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
// and include needed headers
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#endif

/** @brief Value of pi */
constexpr auto pi = 3.14159265358979323846;

/**
 * @brief Degrees to radians
 * @param d Degrees
 * @return Radians
 */
inline float radians(float d)
{
  return ((d) * (pi / 180.0f));
}

/**
 * @brief Distance of 1 Arc Second (longitude, latitude) using the WGS84 Ellipsoid model
 * @see https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84
 * @param lat Latitude where the arc second distance is calculated
 * @return std::tuple<float, float> 1 arc distance in meters (longitude, latitude) for 1 arc sec at the given latitude
 */
inline std::tuple<float, float> arcSecMeters(float lat)
{
  float a{6378137.0f};            /*!< Semi-major axis (a) of the Earth (m) */
  float f = 1 / 298.257223563;    /*!< Flattening factor of the Earth */
  float e2 = (2.0 * f) - (f * f); /*!< Eccentricity squared of the earth's ellipsoid: (2f - f^2)*/

  // Convert latitude to radians
  float latR = radians(lat);

  float sn2 = sinf(latR) * sinf(latR);
  float cs = cosf(latR);

  // Convert 1 arcsec to radians and multiply by the calculated distance
  float arcSecLon = radians(1.0f / 3600.0f) * ((a * cs) / powf(1.0 - e2 * sn2, 1.5f));
  float arcSecLat = radians(1.0f / 3600.0f) * ((a * (1 - e2)) / powf(1.0 - e2 * sn2, 1.5f));

  return {arcSecLon, arcSecLat};
}

/**
 * @brief Calculates the grid cell size in decimal degrees.
 * @param lat Latitude where the resolution is calculated.
 * @param dxM Grid X resolution in meters
 * @param dyM  Grid Y resolution in meters
 * @return std::tuple<float, float> X, Y resolution in decimal degrees
 */
inline std::tuple<float, float> cellSizeDegrees(float lat, float dxM, float dyM)
{

  // Get the distance of 1 arc second in meters at lat
  auto [arcSecLon, arcSecLat] = arcSecMeters(lat);

  // Divide dxM and dyM by the calculated distance of 1 arc second to calculate arcseconds
  // Divide resulting arc seconds by 3600 to convert to decimal degrees
  float dxDeg = (dxM / arcSecLon) / 3600.0f;
  float dyDeg = (dyM / arcSecLat) / 3600.0f;

  return {dxDeg, dyDeg};
}

/**
 * @brief Status of the operation.
 */
enum class status : int
{
  SUCCESS = 0,  /*!< OK */
  FAILURE = -1, /*!< Operation was not successful. */
};

/**
 * @brief Get page size from the operating system
 * @return Page size in bytes
 * @see https://create.stephan-brumme.com/portable-memory-mapping/
 */
inline int getPageSize()
{
#ifdef _MSC_VER
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return sysInfo.dwAllocationGranularity;
#else
  return sysconf(_SC_PAGESIZE);
#endif
}

#endif