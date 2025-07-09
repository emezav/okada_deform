/**
 * @file
 * @brief CUDA implementation of Okada deformation model
 * Okada Y., Surface deformation due to shear and tensile faults in a half-space
 * Bull. Seismol. Soc. Am., 75:4, 1135-1154, 1985.
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 */

#ifndef OKADA85_CU
#define OKADA85_CU

#include "cuda_common.h"
#include "okada85.h"

/**
 * @brief Interface for the GPU version of Okada85
 */
namespace okada85gpu
{
    using namespace okada85;

    /**
     * @brief Deformation due to shear and tensile faults in half-space
     * @param rows Grid height - rows
     * @param columns Grid width - columns
     * @param x0lon Lower-left longitude of the grid system (WGS84)
     * @param y0lat Lower-left latitude of the grid system (WGS84)
     * @param dx Grid point X (East - West) separation (m)
     * @param dy Grid point Y (Up north) separation (m)
     * @param components Fault components
     * @param nComponents Number of fault components
     * @param params Model parameters for the calculation
     * @param Uz (Output) Resulting deformation grid (Z direction - up down) in a single array[width * height]
     * @param Us (Output) Resulting deformation grid (Strike direction) in a single array[width * height]
     * @param Ud (Output) Resulting deformation grid (Dip direction) in single array[width * height]
     * @param Ux (Output) Resulting deformation grid (X direction - longitude) in a single array[width * height]
     * @param Uy (Output) Resulting deformation grid (Y direction - latitude) in a single array[width * height]
     * @note Caller must allocate memory for output parameters.
     */
    __host__ okadaStatus deform(
        int rows,
        int columns,
        float x0lon,
        float y0lat,
        float dx,
        float dy,
        fault *components,
        int nComponents,
        parameters params,
        float *Uz,
        float *Us,
        float *Ud,
        float *Ux,
        float *Uy);

    /**
     * @brief Deformation due to shear and tensile faults in half-space
     * @param h Bathymetry to be deformed
     * @param rows Grid height - rows
     * @param columns Grid width - columns
     * @param x0lon Lower-left longitude of the grid system (WGS84)
     * @param y0lat Lower-left latitude of the grid system (WGS84)
     * @param dx Grid point X (East - West) separation (m)
     * @param dy Grid point Y (Up north) separation (m)
     * @param components Fault components
     * @param nComponents Number of fault components
     * @param params Model parameters for the calculation
     * @param Uz (Output) Resulting deformation grid (Z direction - up down) in a single array[width * height]
     * @param Us (Output) Resulting deformation grid (Strike direction) in a single array[width * height]
     * @param Ud (Output) Resulting deformation grid (Dip direction) in single array[width * height]
     * @param Ux (Output) Resulting deformation grid (X direction - longitude) in a single array[width * height]
     * @param Uy (Output) Resulting deformation grid (Y direction - latitude) in a single array[width * height]
     * @param Ub (Output) Resulting deformation on the bathymetry
     * @note Caller must allocate memory for output parameters.
     */
    __host__ okadaStatus deform(
        float * h,
        int rows,
        int columns,
        float x0lon,
        float y0lat,
        float dx,
        float dy,
        fault *components,
        int nComponents,
        parameters params,
        float *Uz,
        float *Us,
        float *Ud,
        float *Ux,
        float *Uy,
        float * Ub,
        bool invbat=true);
}

#endif