/**
 * @file
 * @brief CUDA implementation of Okada deformation model
 * Okada Y., Surface deformation due to shear and tensile faults in a  half-space
 * Bull. Seismol. Soc. Am., 75:4, 1135-1154, 1985.
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 */

#include <iostream>

#include "geo.h"
#include "okada85gpu.cuh"

namespace okada85gpu
{
    using namespace geo;
    using namespace okada85;
    using std::cerr;
    using std::cout;
    using std::endl;

    using geo::pi;

    /**
     * @brief Parameter to check if a value is close to zero.
     */
    __device__ __constant__ float cuEps;

    /**
     * @brief Count of points on the grid
     */
    __device__ __constant__ int cuNumPoints;

    /**
     * @brief Number of columns on the grid
     */
    __device__ __constant__ int cuColumns;

    /**
     * @brief Number of rows on the grid
     */
    __device__ __constant__ int cuRows;

    /**
     * @brief dx Grid X resolution (longitude) in meters.
     */
    __device__ __constant__ float cuDx;

    /**
     * @brief Grid Y resolution (latitude) in meters.
     */
    __device__ __constant__ float cuDy;

    /**
     * @brief Magnitude of the dislocation parallel to the strike
     */
    __device__ __constant__ float cuU1;

    /**
     * @brief Magnitude of the dislocation perpendicular to the strike
     */
    __device__ __constant__ float cuU2;

    /**
     * @brief Magnitude of the dislocation in the Z axis (up-down)
     */
    __device__ __constant__ float cuU3;

    /**
     * @brief Mu/Lambda coefficient
     */
    __device__ __constant__ float cuMu_L;

    /**
     * @brief Length of the fault plane
     */
    __device__ __constant__ float cuL;

    /**
     * @brief Width of the fault plane
     */
    __device__ __constant__ float cuW;

    /**
     * @brief Cosine of dip
     */
    __device__ __constant__ float cuCs;

    /**
     * @brief Sin of dip
     */
    __device__ __constant__ float cuSn;

    /**
     * @brief Cosine of strike
     */
    __device__ __constant__ float cuCsStr;

    /**
     * @brief Sin of strike
     */
    __device__ __constant__ float cuSnStr;

    /**
     * @brief Tangent of strike
     */
    __device__ __constant__ float cuTnStr;

    /**
     * @brief Checks if the absolute value ot the parameter is less than epsilon
     * @param val Value to check if its close to zero
     * @return True if the absolute value is less than epsilon (close to zero), false otherwise.
     */
    __forceinline__ __device__ bool isZero(float val)
    {
        return (fabsf(val) < cuEps);
    }

    /**
     * @brief Returns the linear position for an element inside a 2D array, row first
     * @param row Row
     * @param column Column
     * @return Linear position
     */
    __forceinline__ __device__ int linRC(int row, int column)
    {
        return (row * cuColumns) + column;
    }

    /**
     * @brief Returns the linear position for an element inside a 2D array, column first
     * @param column Column
     * @param row Row
     * @return Linear position
     */
    __forceinline__ __device__ int linCR(int column, int row)
    {
        return (row * cuColumns) + column;
    }

    /**
     * @brief Calculates the deformation for a fault event.
     * @param i0 Origin of the fault on the grid - column
     * @param j0 Origin of the fault on the grid - row
     * @param de Depth
     * @param Uz Deformation on the Z axis
     * @param Us Deformation on the direction of the strike
     * @param Ud Deformation on the direction of the dip
     * @param Ux Deformation on X
     * @param Uy Deformation on Y
     */
    __global__ void deform_kernel(
        int i0,
        int j0,
        float de,
        float *Uz,
        float *Us,
        float *Ud,
        float *Ux,
        float *Uy);

    /**
     * @brief Apply the calculated deformation on Uz, Ux and Uy to a bathymetry
     *
     * @param h Bathymetry
     * @param Uz Deformation on the Z axis
     * @param Ux Deformation on the X axis
     * @param Uy Deformation on the Y axis
     * @param Ub Calculated deformation on the bathymetry
     * @return __global__
     */
    __global__ void deform_bathymetry_kernel(
        float *h,
        float *Uz,
        float *Ux,
        float *Uy,
        float *Ub,
        bool invbat);

    /**
     * @brief Calculates Chinnery's strike / slip (25) PP. 1144 for (24) PP. 1143
     * @param x Outer integral variable (23) PP. 1143
     * @param p Inner integral variable (23) PP. 1143
     * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
     * @param Ux (Output) Reference to store the result for the x component (26)
     * @param Uy (Output) Reference to store the result for the y component (26)
     * @param Uz (Output) Reference to store the result for the z component (26)
     */
    __noinline__ __device__ void chinneryStrikeSlip(
        float x,
        float p,
        float q,
        float &Ux,
        float &Uy,
        float &Uz);

    /**
     * @brief Calculates Chinnery's (24) PP. 1143 for dip (26) PP. 1144
     * @param x Outer integral variable (23) PP. 1143
     * @param p Inner integral variable (23) PP. 1143
     * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
     * @param Ux (Output) Reference to store the result for the x component (26)
     * @param Uy (Output) Reference to store the result for the y component (27)
     * @param Uz (Output) Reference to store the result for the z component (28)
     */
    __noinline__ __device__ void chinneryDipSlip(
        float x,
        float p,
        float q,
        float &Ux,
        float &Uy,
        float &Uz);

    /**
     * @brief Calculates the inner part [...] for ux, uy, uz for strike-slip given Xi, Eta (25) PP. 1144
     *
     */
    __noinline__ __device__ void strikeSlip(
        float Xi,
        float Eta,
        float p,
        float q,
        float &Fx,
        float &Fy,
        float &Fz);

    /**
     * @brief Calculates the inner part [...] for ux, uy, uz for dip-slip given Xi, Eta (25) PP. 1144
     *
     */
    __noinline__ __device__ void dipSlip(
        float Xi,
        float Eta,
        float p,
        float q,
        float &Fx,
        float &Fy,
        float &Fz);

    /**
     * @brief I1 function (28), (29), PP. 1144, 1145
     * @param Xi Xi from Chinery's notation (24) PP. 1143
     * @param Eta Eta from Chinery's notation (24) PP.1143
     * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
     * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
     * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
     * @param i5 Value of I5(Xi, Eta, ...)
     */
    __noinline__ __device__ float I1(
        float Xi,
        float Eta,
        float R,
        float dTilde,
        float q,
        float i5);

    /**
     * @brief I2 function (28), (29) PP. 1144, 1145
     * @param Xi Xi from Chinery's notation (24) PP. 1143
     * @param Eta Eta from Chinery's notation (24) PP.1143
     * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
     * @param yTilde eta*cos(dip) - q*sin(dip) (30) PP.1143
     * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
     * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
     * @param i3 value of I3(Xi, Eta, ...)
     */
    __noinline__ __device__ float I2(
        float Xi,
        float Eta,
        float R,
        float yTilde,
        float dTilde,
        float q,
        float i3);

    /**
     * @brief I3 function (28), (29) PP. 1144, 1145
     * @param Xi Xi from Chinery's notation (24) PP. 1143
     * @param Eta Eta from Chinery's notation (24) PP.1143
     * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
     * @param yTilde eta*cos(dip) - q*sin(dip) (30) PP.1143
     * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
     * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
     * @param i4 Value of I4(Xi, Eta, ...)
     */
    __noinline__ __device__ float I3(
        float Xi,
        float Eta,
        float R,
        float yTilde,
        float dTilde,
        float q,
        float i4);

    /**
     * @brief I4 function (28), (29) PP. 1144, 1145
     * @param Xi Xi from Chinery's notation (24) PP. 1143
     * @param Eta Eta from Chinery's notation (24) PP.1143
     * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
     * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
     * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
     */

    __noinline__ __device__ float I4(
        float Xi,
        float Eta,
        float R,
        float dTilde,
        float q);
    /**
     * @brief I5 function (28), (29) PP. 1144, 1145
     * @param Xi Xi from Chinery's notation (24) PP. 1143
     * @param Eta Eta from Chinery's notation (24) PP.1143
     * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
     * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
     * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
     */
    __noinline__ __device__ float I5(
        float Xi,
        float Eta,
        float R,
        float dTilde,
        float q);

    __host__ okadaStatus deform(
        float *h,
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
        float *Ub,
        bool invbat)
    {

        cudaError_t cudaStatus;

        okadaStatus status = deform(
            rows,
            columns,
            x0lon,
            y0lat,
            dx,
            dy,
            components,
            nComponents,
            params,
            Uz,
            Us,
            Ud,
            Ux,
            Uy);

        if (status != okadaStatus::SUCCESS)
        {
            return status;
        }

        // Create a Stream for the kernel
        cudaStream_t stream;
        cudaStatus = cudaStreamCreate(&stream);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Unable to create execution stream" << endl;
            return okadaStatus::FAILURE;
        }

        // Launch a kernel on the GPU with one thread for each element.
        dim3 threadsPerBlock(16, 16); // Attempt to use all the 1024
        dim3 blocks((columns / threadsPerBlock.x) + 1, (rows / threadsPerBlock.y) + 1);

        // Calculate the deformation on the grid caused by this fault event.
        // All other parameters have been calculated and sent to the device as constants.
        deform_bathymetry_kernel<<<blocks, threadsPerBlock, 0, stream>>>(h, Uz, Ux, Uy, Ub, invbat);

        // Execution error?
        cudaStatus = cudaGetLastError();

        if (cudaStatus != cudaSuccess)
        {
            cerr << "Failed to launch deform bathymetry kernel : (error code : " << cudaGetErrorString(cudaStatus) << endl;
        }

        /*
         * WAIT FOR ALL THREADS TO FINISH
         * Host code needs to wait until this event is evaluated on all points of the grid.
         * Only after this condition is met, the next fault event can be evaluated.
         */
        cudaStatus = cudaStreamSynchronize(stream);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Error on cudaDeviceSynchronize" << endl;
            cerr << cudaGetErrorString(cudaStatus) << " " << __FILE__ << " " << __LINE__ << endl;
            cerr << "Error launching kernel" << endl;
            return okadaStatus::FAILURE;
        }

        return okadaStatus::SUCCESS;
    }

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
        float *Uy)
    {

        cudaError_t cudaStatus;

        // Get deformation parameters
        auto [vp, vs, re, e2, Eps] = params;

        // Proxy variables
        float vs2 = vs * vs; // S-wave speed
        float vp2 = vp * vp; // P-wave speed

        // Mu_L Coefficient for I1 - I5 (28), (29)
        float Mu_L = vs2 / (vp2 - vs2);

        // Calculate the distance of 1 arc second at the latitude
        // of the origin of the grid
        auto [xdst, ydst] = arcSecMeters(y0lat);

        // drx - dx in arc seconds
        float drx = dx / xdst;

        // dry - dy in arc seconds
        float dry = dy / ydst;

        /*
         * COPY MODEL CONSTANTS TO DEVICE
         * These variables become "constants" for all points on the grid, and to all kernels:
         * cs: cos(dip)
         * sn: sin(dip)
         * csStr: cos(strike)
         * snStr: sin(strike)
         * tnStr: tan(strike)
         */

        // Eps constant
        cudaStatus = cudaMemcpyToSymbol((const void *)&cuEps, &Eps, sizeof(float), 0, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Failed to copy constant Eps from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
            return okadaStatus::FAILURE;
        }

        // Mu/L constant
        cudaStatus = cudaMemcpyToSymbol((const void *)&cuMu_L, &Mu_L, sizeof(float), 0, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Failed to copy constant Mu_L from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
            return okadaStatus::FAILURE;
        }

        // Grid columns
        cudaStatus = cudaMemcpyToSymbol((const void *)&cuColumns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Failed to copy constant columns from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
            return okadaStatus::FAILURE;
        }

        // Grid rows
        cudaStatus = cudaMemcpyToSymbol((const void *)&cuRows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Failed to copy constant rows from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
            return okadaStatus::FAILURE;
        }

        // Grid X resolution
        cudaStatus = cudaMemcpyToSymbol((const void *)&cuDx, &dx, sizeof(float), 0, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Failed to copy constant dx from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
            return okadaStatus::FAILURE;
        }

        // Grid Y resolution
        cudaStatus = cudaMemcpyToSymbol((const void *)&cuDy, &dy, sizeof(float), 0, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Failed to copy constant dy from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
            return okadaStatus::FAILURE;
        }

        // Create a Stream for the kernels
        cudaStream_t stream;
        cudaStatus = cudaStreamCreate(&stream);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Unable to create execution stream" << endl;
            return okadaStatus::FAILURE;
        }

        // Launch a kernel on the GPU with one thread for each element.
        dim3 threadsPerBlock(16, 16); // Attempt to use all the 1024
        dim3 blocks((columns / threadsPerBlock.x) + 1, (rows / threadsPerBlock.y) + 1);

        // cout << "Blocks: " << blocks.x << " x " << blocks.y << endl;
        // cout << "Threads per block: " << threadsPerBlock.x << " x " << threadsPerBlock.y << endl;
        // cout << "Total threads: " << (blocks.x * threadsPerBlock.x) * (blocks.y * threadsPerBlock.y) << endl;

        // For each one of the components
        for (int n = 0; n < nComponents; n++)
        {
            // Get parameters from the fault
            auto [lon, lat, d, length, width, st, di, sl, hh] = components[n];

            // Convert strike, dip and slip angles to radians
            float str = radians(90.0f - st); // st (north up) to rad
            float dir = radians(di);         // dip to rad
            float slr = radians(sl);         // slip to rad

            // Calculate sin and cos of dip angle
            float cs = cosf(dir);
            float sn = sinf(dir);

            // Calculate cos, sin an tan of strike angle
            float csStr = cosf(str);
            float snStr = sinf(str);
            float tnStr = tanf(str);

            // Calculate distance from height and strike angle
            float de = hh + width * sn;

            // Calculate U1, U2, and U3 components on the Okada coordiante system
            float U1 = d * cosf(slr);
            float U2 = d * sinf(slr);
            float U3 = d; // Dislocation

            // Calculate i, j fault position on the grid relative to the grid origin
            int i0 = (lon - x0lon) * 3600.0f / drx;
            int j0 = (lat - y0lat) * 3600.0f / dry;

            /*
             * COPY EVENT CONSTANTS TO DEVICE
             * These variables become "constants" for all points on the grid, but need to be set for each fault event.
             *
             * Because each fault event needs to be evaluated before the next one (meaning the effect of this fault
             * needs to be evaluated on all points on the grid before the next event is evaluated), the constants
             * must be right before calling the device code.
             *
             * cs: cos(dip)
             * sn: sin(dip)
             * csStr: cos(strike)
             * snStr: sin(strike)
             * tnStr: tan(strike)
             */

            // cos of dip
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuCs, &cs, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant cs from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // sin of dip
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuSn, &sn, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant sn from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // cos of strike
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuCsStr, &csStr, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant csStr from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // sin of strike
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuSnStr, &snStr, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant snStr from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // tan of strike
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuTnStr, &tnStr, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant tnStr from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // Fault length
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuL, &length, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant length from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // Fault width
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuW, &width, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant width from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // U1 component
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuU1, &U1, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant U1 from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // U2 component
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuU2, &U2, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant U2 from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // U3 component
            cudaStatus = cudaMemcpyToSymbol((const void *)&cuU3, &U3, sizeof(float), 0, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to copy constant U3 from host to device (error code " << cudaGetErrorString(cudaStatus) << ")" << endl;
                return okadaStatus::FAILURE;
            }

            // Calculate the deformation on the grid caused by this fault event.
            // All other parameters have been calculated and sent to the device as constants.
            deform_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
                i0,
                j0,
                de,
                Uz,
                Us,
                Ud,
                Ux,
                Uy);

            // Execution error?
            cudaStatus = cudaGetLastError();

            if (cudaStatus != cudaSuccess)
            {
                cerr << "Failed to launch deform kernel for the " << n << "th fault event : (error code : " << cudaGetErrorString(cudaStatus) << endl;
            }

            /*
             * WAIT FOR ALL THREADS TO FINISH
             * Host code needs to wait until this event is evaluated on all points of the grid.
             * Only after this condition is met, the next fault event can be evaluated.
             */
            cudaStatus = cudaStreamSynchronize(stream);
            if (cudaStatus != cudaSuccess)
            {
                cerr << "Error on cudaDeviceSynchronize" << endl;
                cerr << cudaGetErrorString(cudaStatus) << " " << __FILE__ << " " << __LINE__ << endl;
                cerr << "Error launching kernel" << endl;
                return okadaStatus::FAILURE;
            }
        }

        return okadaStatus::SUCCESS;
    }

    /** ============ Device kernels implementation ==================== */

    __global__ void deform_kernel(
        int i0,
        int j0,
        float de,
        float *Uz,
        float *Us,
        float *Ud,
        float *Ux,
        float *Uy)
    {

        // Calculate this thread position on the grd
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        // If this thread position is inside the grid
        if (i < cuColumns && j < cuRows)
        {
            // Transform into Okada coordinate system.

            /*
             * Text fragment from PP. 1138
             * Elastic medium occupies the region of z =<0 and x axis is taken to be parallel to the strike
             * direction of the fault. Further, we define elementary dislocations U1, U2, and U3 so
             * as to correspond to strike-slip, dip-slip, and tensile components of arbitrary dislocation.
             */

            // X0, y0: Position of the point grid (i, j) relative to i0, j0 in meters
            float x0 = (i - i0) * cuDx;
            float y0 = (j - j0) * cuDy;

            // Transform into Okada's coordinate system
            float x = x0 / cuCsStr + (y0 - x0 * cuTnStr) * cuSnStr;
            float y = y0 * cuCsStr - x0 * cuSnStr + cuW * cuCs;

            // Definitions for p and q from (30) PP. 1145
            float p = y * cuCs + de * cuSn;
            float q = y * cuSn - de * cuCs;

            // Calculate the strike components for ux, uy and uz (25) PP. 1144 using Chinnery's notation f(Xi, Eta)
            float uxStr, uyStr, uzStr;
            chinneryStrikeSlip(x, p, q, uxStr, uyStr, uzStr);

            // Calculate the dip components for ux, uy and uz (26) PP. 1144 using Chinnery's notation f(Xi, Eta)
            float uxDip, uyDip, uzDip;
            chinneryDipSlip(x, p, q, uxDip, uyDip, uzDip);

            // NOTE: Tensile components (27) PP.1144 aren't calculated.
            // Implement yourself, or contact me in case you need help!

            // Calculate this thread offset on the 1D flattened array
            // (row * columns) + column
            // int pos = (j * cuColumns) + i;
            int pos = linCR(i, j);

            // Add to Uz
            Uz[pos] += uzStr + uzDip;

            // Add to Us
            Us[pos] += uxStr + uxDip;

            // Add to Ud
            Ud[pos] += uyStr + uyDip;

            // Calculate Ux
            Ux[pos] = Us[pos] * cuCsStr - Ud[pos] * cuSnStr;
            // Ux[pos] = (j * cuColumns) + i;

            // Calculate Uy
            Uy[pos] = Us[pos] * cuSnStr + Ud[pos] * cuCsStr;

            // Discard small deformations
            if (fabs(Uz[pos]) <= 0.01f)
            {
                Uz[pos] = 0.0f;
            }
            if (fabs(Us[pos]) <= 0.01f)
            {
                Us[pos] = 0.0f;
            }
            if (fabs(Ud[pos]) <= 0.01f)
            {
                Ud[pos] = 0.0f;
            }
            if (fabs(Ux[pos]) <= 0.01f)
            {
                Ux[pos] = 0.0f;
            }
            if (fabs(Uy[pos]) <= 0.01f)
            {
                Uy[pos] = 0.0f;
            }
        } // If this thread is inside the grid
    }

    /**
     * @brief Apply the calculated deformation on Uz, Ux and Uy to a bathymetry
     *
     * @param h Bathymetry
     * @param Uz Deformation on the Z axis
     * @param Ux Deformation on the X axis
     * @param Uy Deformation on the Y axis
     * @param Ub Calculated deformation on the bathymetry
     * @return __global__
     */
    __global__ void deform_bathymetry_kernel(
        float *h,
        float *Uz,
        float *Ux,
        float *Uy,
        float *Ub,
        bool invbat)
    {
        // Calculate this thread position on the grd
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        float factor = 1.0f;
        if (invbat)
        {
            factor = -1.0f;
        }

        // If this thread position is inside the grid
        if (i < cuColumns && j < cuRows)
        {
            // Calculate this thread offset on the 1D flattened array
            // (row * columns) + column
            // int pos = (j * cuColumns) + i;
            int pos = linCR(i, j);

            // Default to zero
            Ub[pos] = 0.0f;

            if (i >= 1 && i < cuColumns - 1 && j >= 1 && j < cuRows - 1)
            {

                int posLeft = linCR(i - 1, j);
                int posRight = linCR(i + 1, j);
                int posBelow = linCR(i, j + 1);
                int posAbove = linCR(i, j - 1);

                int hPos = h[pos] * factor;
                int hRight = h[posRight] * factor;
                int hLeft = h[posLeft] * factor;
                int hBelow = h[posBelow] * factor;
                int hAbove = h[posAbove] * factor;

                hPos = (hPos <= 0.0f) ? 0.0f : hPos;
                hRight = (hRight <= 0.0f) ? 0.0f : hRight;
                hLeft = (hLeft <= 0.0f) ? 0.0f : hLeft;
                hBelow = (hBelow <= 0.0f) ? 0.0f : hBelow;
                hAbove = (hAbove <= 0.0f) ? 0.0f : hAbove;

                Ub[pos] = Uz[pos] +
                          Ux[pos] * (hRight - hLeft) / (2.0f * cuDx) +
                          Uy[pos] * (hBelow - hAbove) / (2.0f * cuDy);

                // Calculate surface deformation, only if bathymetry is greater than zero
                if ((isZero(hRight) && isZero(hLeft)) ||
                    (isZero(hBelow) && isZero(hAbove)))
                {
                    Ub[pos] = Uz[pos];
                }
            }
        }
    }

    __noinline__ __device__ void chinneryStrikeSlip(
        float x,
        float p,
        float q,
        float &Ux,
        float &Uy,
        float &Uz)
    {

        // Calculates Chinnery's (24) PP. 1143 for strike-slip (25) PP. 1144
        // f(Xi, Eta) = f(x, p) - f(x, p - W) - f(x-L, p) + f(x - L, p - W)
        // Strike must be evaluated for 4 values of Xi, Eta
        // and calculates x, y, z component for each f.
        // Xi = x,     Eta = p      (f1): calculates fx1, fy1, fz1
        // Xi = x,     Eta = p - W  (f2): calculates fx2, fy2, fz2
        // Xi = x - L, Eta = p      (f3): calculates fx3, fy3, fz3
        // Xi = x - L, Eta = p - W  (f4): calculates fx4, fy4, fz4

        // X component of strike-slip to replace on (24)
        float fx1{0};
        float fx2{0};
        float fx3{0};
        float fx4{0};

        // Y component of strike-slip to replace on (24)
        float fy1{0};
        float fy2{0};
        float fy3{0};
        float fy4{0};

        // Z component of strike-slip to replace on (24)
        float fz1{0};
        float fz2{0};
        float fz3{0};
        float fz4{0};

        // f1: strikeSlip(x, p) (25) PP. 1144
        strikeSlip(x, p, p, q, fx1, fy1, fz1);
        // f2: strikeSlip(x, p - W) (25) PP. 1144
        strikeSlip(x, p - cuW, p, q, fx2, fy2, fz2);
        // f3: strikeSlip(x -L, p) (25) PP. 1144
        strikeSlip(x - cuL, p, p, q, fx3, fy3, fz3);
        // f4: strikeSlip(x -L, p - W) (25) PP. 1144
        strikeSlip(x - cuL, p - cuW, p, q, fx4, fy4, fz4);

        // Evaluate chinnery notation for components, x, y and z
        Ux = -(cuU1 / (2.0f * pi)) * (fx1 - fx2 - fx3 + fx4);
        Uy = -(cuU1 / (2.0f * pi)) * (fy1 - fy2 - fy3 + fy4);
        Uz = -(cuU1 / (2.0f * pi)) * (fz1 - fz2 - fz3 + fz4);
    }

    __noinline__ __device__ void chinneryDipSlip(
        float x,
        float p,
        float q,
        float &Ux,
        float &Uy,
        float &Uz)
    {
        // Calculates Chinnery's (24) PP. 1143 for dip-slip  (26) PP. 1144
        // f(Xi, Eta) = f(x, p) - f(x, p - W) - f(x-L, p) + f(x - L, p - W)
        // Strike must be evaluated for 4 values of Xi, Eta
        // and calculates x, y, z component for each f.
        // Xi = x,     Eta = p      (f1): calculates fx1, fy1, fz1
        // Xi = x,     Eta = p - W  (f2): calculates fx2, fy2, fz2
        // Xi = x - L, Eta = p      (f3): calculates fx3, fy3, fz3
        // Xi = x - L, Eta = p - W  (f4): calculates fx4, fy4, fz4

        // X component of dip-slip to replace on (24)
        float fx1{0};
        float fx2{0};
        float fx3{0};
        float fx4{0};

        // Y component of dip-slip to replace on (24)
        float fy1{0};
        float fy2{0};
        float fy3{0};
        float fy4{0};

        // Z component of dip-slip to replace on (24)
        float fz1{0};
        float fz2{0};
        float fz3{0};
        float fz4{0};

        // f1: dipSlip(x, p) (26) PP. 1144
        dipSlip(x, p, p, q, fx1, fy1, fz1);
        // f2: dipSlip(x, p - W) (26) PP. 1144
        dipSlip(x, p - cuW, p, q, fx2, fy2, fz2);
        // f3: dipSlip(x -L, p) (26) PP. 1144
        dipSlip(x - cuL, p, p, q, fx3, fy3, fz3);
        // f4: dipSlip(x -L, p - W) (26) PP. 1144
        dipSlip(x - cuL, p - cuW, p, q, fx4, fy4, fz4);

        // Evaluate chinnery notation for components, x, y and z
        Ux = -(cuU2 / (2.0f * pi)) * (fx1 - fx2 - fx3 + fx4);
        Uy = -(cuU2 / (2.0f * pi)) * (fy1 - fy2 - fy3 + fy4);
        Uz = -(cuU2 / (2.0f * pi)) * (fz1 - fz2 - fz3 + fz4);
    }

    __noinline__ __device__ void strikeSlip(
        float Xi,
        float Eta,
        float p,
        float q,
        float &Fx,
        float &Fy,
        float &Fz)

    {

        // Calculates the inner part [...] for ux, uy, uz for strike-slip on given Xi, Eta (25) PP. 1144

        // Singularities when: (PP. 1148)
        // q -> 0: set atanf((Xi * Eta)/(q * R)) to 0 in equation (25)
        // Xi -> 0: set I5 = 0 (handled by I5)
        // (R + Eta) -> 0 all terms which contain (R + Eta) on denominator are set to 0
        // and replace ln(R + Eta) to -ln(R + Eta) in equations (28)

        // Calculate yTilde (30) PP. 1145
        float yTilde = Eta * cuCs + q * cuSn;

        // Calculate dTilde (30) PP. 1145
        float dTilde = Eta * cuSn - q * cuCs;

        // Calculate R (30) PP. 1145
        float R = sqrtf(Xi * Xi + Eta * Eta + q * q);

        // Calculate I1 - I5, (28) (29) PP. 1145
        // Order of calculation: I5, I4, I3 (uses I4), I2 (uses I3), I1 (Uses I5)
        // Each function checks for singularities and return 0 if calculation is not possible.

        float i5 = I5(Xi, Eta, R, dTilde, q);
        float i4 = I4(Xi, Eta, R, dTilde, q);
        float i3 = I3(Xi, Eta, R, yTilde, dTilde, q, i4);
        float i2 = I2(Xi, Eta, R, yTilde, dTilde, q, i3);
        float i1 = I1(Xi, Eta, R, dTilde, q, i5);

        // Fx, Fy, Fz: inner part of equation [ ... ]|| (25) PP. 1144 for ux, uy and uz

        // Initialize Fx, Fy, Fz to the last term of (25) PP. 1144 to return early
        // in case of zero or singularity
        // Fx : I1 * sn
        // Fy : I2 * sn
        // Fz : I4 * sn
        // I1, I2 and I4 return zero if any singularity exists.
        Fx = i1 * cuSn;
        Fy = i2 * cuSn;
        Fz = i4 * cuSn;

        // Terms are evaluated to zero or singularity exists:
        // If q is zero, first and second terms of Fx, Fy and Fz are set to zero:
        // On Fx, q sets the numerator to zero on the first term and singularity on the second term.
        // On Fy and Fz, q sets the numerator to zero on both first and second term.
        if (isZero(q))
        {
            return;
        }

        // Flags to check singularities on R and (R + Eta)
        bool zeroR = isZero(R);
        bool zeroREta = isZero(R + Eta);

        // If Both R and (R + Eta) are not zero, add first terms to Fx, Fy and Fz
        if (!zeroR && !zeroREta)
        {
            // First term of Fx
            Fx += (Xi * q) / (R * (R + Eta));
            // First term of Fy
            Fy += (yTilde * q) / (R * (R + Eta));
            // First term of Fz
            Fz += (dTilde * q) / (R * (R + Eta));
        }

        // If R is not zero (q already checked), add the second term to Fx
        if (!zeroR)
        {
            Fx += atanf((Xi * Eta) / (q * R));
        }

        // If  (R + Eta) is not zero, add second term to Fy and Fz
        if (!zeroREta)
        {
            Fy += (q * cuCs) / (R + Eta);
            Fz += (q * cuSn) / (R + Eta);
        }
    }

    __noinline__ __device__ void dipSlip(
        float Xi,
        float Eta,
        float p,
        float q,
        float &Fx,
        float &Fy,
        float &Fz)
    {
        // Calculates the inner part [...] for ux, uy, uz for dip-slip on given Xi, Eta (25) PP. 1144

        // Singularities when: (PP. 1148)
        // q -> 0: set atanf((Xi * Eta)/(q * R)) to 0 in equation (26)
        // Xi -> 0: set I5 = 0 (handled by I5)
        // (R + Eta) -> 0 all terms which contain (R + Eta) on denominator are set to 0
        // and replace ln(R + Eta) to -ln(R + Eta) in equations (28)

        // Calculate R (30) PP. 1145
        float R = sqrtf(Xi * Xi + Eta * Eta + q * q);

        // Calculate yTilde (30) PP. 1145
        float yTilde = Eta * cuCs + q * cuSn;

        // Calculate dTilde (30) PP. 1145
        float dTilde = Eta * cuSn - q * cuCs;

        // Calculate I1 - I5, (28) (29) PP. 1145
        // Order of calculation: I5, I4, I3 (uses I4), I2 (uses I3), I1 (Uses I5)
        // Each function checks for singularities and return 0 if calculation is not possible.

        float i5 = I5(Xi, Eta, R, dTilde, q);
        float i4 = I4(Xi, Eta, R, dTilde, q);
        float i3 = I3(Xi, Eta, R, yTilde, dTilde, q, i4);
        float i2 = I2(Xi, Eta, R, yTilde, dTilde, q, i3);
        float i1 = I1(Xi, Eta, R, dTilde, q, i5);

        // Fx, Fy, Fz: inner part of equation [ ... ]|| (26) PP. 1144 for ux, uy and uz

        // Initialize Fx, Fy, Fz to the last term of (26) PP. 1144 to return early
        // in case of zero or singularity
        // Fx : -I3 * sn * cs
        // Fy : -I1 * sn * cs
        // Fz : -I5 * sn * cs

        Fx = -i3 * cuSn * cuCs;
        Fy = -i1 * cuSn * cuCs;
        Fz = -i5 * cuSn * cuCs;

        // Terms are evaluated to zero or singularity exists:
        // If q is zero, first and second terms of Fx, Fy and Fz are set to zero:
        // On Fx, q sets the numerator to zero on the first term.
        // On Fy and Fz, q sets the numerator to zero on both first and second term.
        if (isZero(q))
        {
            return;
        }

        // Flags to check singularities on R and (R + Eta)
        bool zeroR = isZero(R);
        bool zeroRXi = isZero(R + Xi);

        if (!zeroR)
        {
            // Add first term of Fx
            Fx += q / R;
            if (!zeroRXi)
            {
                // Add first terms to Fy, Fz
                Fy += (yTilde * q) / (R * (R + Xi));
                Fz += (dTilde * q) / (R * (R + Xi));
            }

            // Add second terms to Fy, Fz
            Fy += cuCs * atanf((Xi * Eta) / (q * R));
            Fz += cuSn * atanf((Xi * Eta) / (q * R));
        }
    }

    __noinline__ __device__ float I1(
        float Xi,
        float Eta,
        float R,
        float dTilde,
        float q,
        float i5)
    {
        // I1 (28), (29), PP. 1144, 1145
        // Singularities when cs -> 0, (R + dTilde) -> 0

        // Set to zero cover base case, when cs -> 0 && (R + dTilde) -> 0
        float i1{0.0f};

        // Flags to check singularities
        bool zeroCs = isZero(cuCs);
        bool zeroRdTilde = isZero(R + dTilde);

        // Base case: cs -> 0 and (R + dTilde) -> 0
        if (zeroCs && zeroRdTilde)
        {
            return i1;
        }

        // POST: cs is not zero or (R + dTilde) is not zero

        // I5 factor only is calculated if cs is not zero
        float i5Factor = (zeroCs ? 0.0f : (cuSn / cuCs) * i5);

        // When cs -> 0, use  (29), otherwise use (28). (R + dTilde) -> 0 already covered by base case.
        i1 = (zeroCs ? -(cuMu_L / 2.0f) * ((Xi * q) / ((R + dTilde) * (R + dTilde)))
                     : cuMu_L * (-Xi / (cuCs * (R + dTilde))));

        // Substract i5Factor (or zero)
        i1 -= i5Factor;

        return i1;
    }

    __noinline__ __device__ float I2(
        float Xi,
        float Eta,
        float R,
        float yTilde,
        float dTilde,
        float q,
        float i3)
    {
        // I2 (28), (29) PP. 1144, 1145
        // Singularity when (R + Xi)

        // Set to zero
        float i2{0.0f};

        // (28) PP.1144. Replace ln(R + Eta) to -ln(R + Eta) when (R + Eta) -> 0
        i2 = cuMu_L * ((R + Eta) >= cuEps ? (-logf(R + Eta)) : (logf(R - Eta))) - i3;

        return i2;
    }

    __noinline__ __device__ float I3(
        float Xi,
        float Eta,
        float R,
        float yTilde,
        float dTilde,
        float q,
        float i4)
    {
        // I3 (28), (29) PP. 1144, 1145

        // Singularities when
        // cs -> 0, (R + dTilde) -> 0, (R + Eta) -> 0

        float i3{0.0f};

        bool zeroCs = isZero(cuCs);
        bool zeroRdTilde = isZero(R + dTilde);
        bool zeroREta = isZero(R + Eta);

        // When all singularities exist, return zero.
        if (zeroCs && zeroRdTilde && zeroREta)
        {
            return i3;
        }

        // If cs is zero, use (29) PP. 1145
        if (zeroCs)
        {
            // Check for singularities on (R + dTilde) and (R + Eta)

            // If (R + dTilde is not zero), add first two terms on I3 (29) PP. 1145
            if (!zeroRdTilde)
            {
                i3 += (Eta / (R + dTilde)) + ((yTilde * q) / ((R + dTilde) * (R + dTilde)));
            }

            // Substract third term on (29) PP. 1145
            // If (R + Eta) is not zero substract last term.
            if (!zeroREta)
            {
                i3 -= logf(R + Eta);
            }
            else
            {
                // If (R + Eta) -> 0, replace ln(R + Eta) to - ln(R - Eta)
                // Substraction of a negative, change to addition
                i3 += logf(R - Eta);
            }
            // Multiply bf Mu_l / 2
            i3 *= (cuMu_L / 2.0f);
        }
        else
        {
            // cs is not zero, use (28) PP. 1144
            // Check for singularities on (R + dTilde) and (R + Eta)
            // cs -> 0 already covered on "if" block above.

            // Add (set) first term if there is no singularity
            if (!zeroRdTilde)
            {
                i3 += yTilde / (cuCs * (R + dTilde));
            }

            // Substract second term on (28) PP. 1144
            // If (R + Eta) substract last term.
            if (!zeroREta)
            {
                i3 -= logf(R + Eta);
            }
            else
            {
                // If (R + Eta) -> 0, replace ln(R + Eta) to - ln(R - Eta)
                // Substraction of a negative, change to addition
                i3 += logf(R - Eta);
            }

            // Multiply by Mu_L
            i3 *= cuMu_L;

            // Add last term, cs -> 0 already covered!
            i3 += (cuSn / cuCs) * I4(Xi, Eta, R, dTilde, q);
        }

        return i3;
    }

    __noinline__ __device__ float I4(
        float Xi,
        float Eta,
        float R,
        float dTilde,
        float q)
    {
        // I4 (28), (29) PP. 1144, 1145
        // Singularities when cs -> 0, (R + dTilde) -> 0, (R + eta) -> 0

        // Set to zero
        float i4{0.0f};

        // Check flags for singularities
        bool zeroCs = isZero(cuCs);
        bool zeroRdTilde = isZero(R + dTilde);
        bool zeroREta = isZero(R + Eta);

        // Base case, cs -> 0 && (R + dTilde) -> 0 && (R + eta) -> 0
        if (zeroCs && zeroRdTilde && zeroREta)
        {
            return i4;
        }

        // If cs -> 0, use (29) PP. 1145
        if (zeroCs)
        {
            // If (R + dTilde) is not zero, calculate (29) PP. 1114
            if (!zeroRdTilde)
            {
                i4 = -cuMu_L * (q / (R + dTilde));
                return i4;
            }
        }
        else
        {
            // Use (28) PP. 1144

            // Add first term inside [ ... ] to i4
            // cs already checked
            if (!zeroRdTilde)
            {
                // if (R + dTilde) is not zero, add ln(R + dTilde)
                i4 += logf(R + dTilde);
            }
            else
            {
                // (R + dTilde) -> 0, replace ln(R + dTilde) to -ln(R - dTilde)
                i4 -= logf(R - dTilde);
            }

            // Substract second term inside [ ... ] to i4
            if (!zeroREta)
            {
                // If (R + Eta) is not zero, substract
                i4 -= cuSn * logf(R + Eta);
            }
            else
            {
                // If (R + Eta) -> 0, replace ln(R + Eta) to -ln(R + Eta)
                // Addition, not substraction!
                i4 += cuSn * logf(R - Eta);
            }

            // Multiply by Mu_L and 1/cs
            i4 *= cuMu_L * (1.0f / cuCs);
        }

        return i4;
    }

    __noinline__ __device__ float I5(
        float Xi,
        float Eta,
        float R,
        float dTilde,
        float q)
    {
        // I5 (28), (29) PP. 1144, 1145

        // Set to zero to cover base cases:
        // Xi -> 0
        // Xi -> 0 && cs -> 0
        float i5{0.0f};

        // Check flags for singularities
        bool zeroXi = isZero(Xi);
        bool zeroCs = isZero(cuCs);
        bool zeroRdTilde = isZero(R + dTilde);

        // Base case, Xi -> 0, set I5 to zero. (PP. 1148), because:
        // Xi -> 0 causes singularity on I5 (28) and sets I5 (29) to zero.
        // Both Xi -> 0 and cs  -> 0 causes singularity on I5 (28) and sets I5 (29) to zero.
        if (zeroXi)
        {
            return i5;
        }

        // Check for singularity on cs
        if (zeroCs)
        {
            // cs -> 0, use (29) PP. 1145
            i5 = (zeroRdTilde ? 0.0f
                              : -cuMu_L * ((Xi * cuSn) / (R + dTilde)));
        }
        else
        {
            // cs is not zero, use (28) PP. 1144
            float X = sqrt(Xi * Xi + q * q);
            i5 = isZero(R + X) ? 0.0f
                               : ((cuMu_L * 2.0) / cuCs) * atanf(((Eta * (X + (q * cuCs))) + (X * (R + X) * cuSn)) / (Xi * (R + X) * cuCs));
        }

        return i5;
    }
}