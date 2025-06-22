/**
 * @file
 * @brief Implementation of the Okada deformation model in half-space
 * Okada Y., Surface deformation due to shear and tensile faults in a half-space
 * Bull. Seismol. Soc. Am., 75:4, 1135-1154, 1985.
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 */

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Env.h"
#include "geo.h"
#include "okada85cpu.h"
#include "okada85gpu.cuh"
#include "Timer.h"

using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::tuple;

using namespace okada85;

namespace fs = std::filesystem;

enum class status : int
  {
    SUCCESS = 0,  /*!< OK */
    FAILURE = -1, /*!< Operation was not successful. */
  };


/**
 * @brief Allocates host memory for the result grids
 * @param numElements number of elements (rows * columns)
 * @return [status, Uz, Us, Ud, Ux, Ux, Uy] arrays to hold deformation results for each point on the grid
 */
std::tuple<status, float *, float *, float *, float *, float *> allocateHostGrids(int numElements)
{
    // Allocate memory for the grids

    // Allocate and zero
    float *Uz = (float *)calloc(numElements, sizeof(float));

    if (Uz == NULL)
    {
        cerr << "Unable to allocate host memory for Uz" << endl;
        return {status::FAILURE, NULL, NULL, NULL, NULL, NULL};
    }

    // Allocate and zero
    float *Us = (float *)calloc(numElements, sizeof(float));

    if (Us == NULL)
    {
        cerr << "Unable to allocate host memory for Us" << endl;
        // Release memory allocated so far
        free(Uz);
        return {status::FAILURE, NULL, NULL, NULL, NULL, NULL};
    }

    // Allocate and zero
    float *Ud = (float *)calloc(numElements, sizeof(float));

    if (Ud == NULL)
    {
        cerr << "Unable to allocate host memory for Ud" << endl;
        // Release memory allocated so far
        free(Uz);
        free(Us);
        return {status::FAILURE, NULL, NULL, NULL, NULL, NULL};
    }

    // Allocate and zero
    float *Ux = (float *)calloc(numElements, sizeof(float));

    if (Ux == NULL)
    {
        cerr << "Unable to allocate host memory for Ux" << endl;
        // Release memory allocated so far
        free(Uz);
        free(Us);
        free(Ud);
        return {status::FAILURE, NULL, NULL, NULL, NULL, NULL};
    }

    // Allocate and zero
    float *Uy = (float *)calloc(numElements, sizeof(float));

    if (Uy == NULL)
    {
        cerr << "Unable to allocate host memory for Ux" << endl;
        // Release memory allocated so far
        free(Uz);
        free(Us);
        free(Ud);
        free(Ux);
        return {status::FAILURE, NULL, NULL, NULL, NULL, NULL};
    }

    return {status::SUCCESS, Uz, Us, Ud, Ux, Uy};
}

/**
 * @brief Allocates device memory for the result grids
 * @param Uz Pointer to Z deformation grid (Z - axis)
 * @param Us Pointer to deformation grid on the strike direction
 * @param Ud Pointer to deformation grid on the dip direction
 * @param Ux Pointer to deformation grid on the X (longitude) directrion
 * @param Uy Pointer to deformation grid on the Y (latitude) directrion
 * @param numElements number of elements (rows * columns)
 * @return [status, Uz, Us, Ud, Ux, Ux, Uy] arrays to hold deformation results for each point on the grid
 */
std::tuple<status, float *, float *, float *, float *, float *> allocateDeviceGrids(
    float *Uz,
    float *Us,
    float *Ud,
    float *Ux,
    float *Uy,
    int numElements)
{
    // Allocate memory for the grids
    float *d_Uz;
    float *d_Us;
    float *d_Ud;
    float *d_Ux;
    float *d_Uy;

    cudaError_t cudaStatus;

    // cudaError_t cudaStatus = allocateDeviceMemory(&d_Uz, numElements);
    cudaStatus = allocateAndCopyToDevice(&d_Uz, Uz, numElements);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "Unable to allocate device memory for Uz: " << cudaGetErrorString(cudaStatus) << endl;
        return {
            status::FAILURE,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
        };
    }

    cudaStatus = allocateAndCopyToDevice(&d_Us, Us, numElements);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "Unable to allocate device memory for Us: " << cudaGetErrorString(cudaStatus) << endl;
        // Release memory allocated so far
        cudaFree(d_Uz);
        return {
            status::FAILURE,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
        };
    }

    cudaStatus = allocateAndCopyToDevice(&d_Ud, Ud, numElements);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "Unable to allocate device memory for Ud: " << cudaGetErrorString(cudaStatus) << endl;
        // Release memory allocated so far
        cudaFree(d_Uz);
        cudaFree(d_Us);
        return {
            status::FAILURE,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
        };
    }

    cudaStatus = allocateAndCopyToDevice(&d_Ux, Ux, numElements);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "Unable to allocate device memory for Ux: " << cudaGetErrorString(cudaStatus) << endl;
        // Release memory allocated so far
        cudaFree(d_Uz);
        cudaFree(d_Us);
        cudaFree(d_Ud);
        return {
            status::FAILURE,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
        };
    }

    cudaStatus = allocateAndCopyToDevice(&d_Uy, Uy, numElements);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "Unable to allocate device memory for Uy: " << cudaGetErrorString(cudaStatus) << endl;
        // Release memory allocated so far
        cudaFree(d_Uz);
        cudaFree(d_Us);
        cudaFree(d_Ud);
        cudaFree(d_Ux);
        return {
            status::FAILURE,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
        };
    }

    return {status::SUCCESS, d_Uz, d_Us, d_Ud, d_Ux, d_Uy};
}

/**
 * @brief Releases host memory for the supplied pointers
 * @param  pointers Vector of pointers to free
 */
void releaseHostMemory(std::vector<void *> pointers)
{
    for (auto ptr : pointers)
    {
        free(ptr);
    }
}

/**
 * @brief Releases device memory for the supplied pointers
 * @param  pointers Vector of pointers to free
 */
void releaseDeviceMemory(std::vector<void *> pointers)
{
    for (auto ptr : pointers)
    {
        cudaFree(ptr);
    }
}

/**
 * @brief Loads the configuration from a scenario file
 * @param path Scenario configuration file path
 * @return std::tuple<status, float, float, int, int, float, float, fault *, int> Scenario configuration
 * @see README.md contains documentation about the format of the scenario file
 */
std::tuple<status, float, float, int, int, float, float, fault *, int> loadScenario(const char *path)
{

    float x0ll{}, y0ll{}, dx{}, dy{};
    int rows{}, columns{};

    vector<fault> faults{};
    fault *faultComponents;
    int nCompoments;

    // Open file for reading
    FILE *fp = fopen(path, "r");

    if (fp == NULL)
    {
        cerr << "Could not open fault parameters from file " << endl;
        return {status::FAILURE, {}, {}, {}, {}, {}, {}, {NULL}, {}};
    }

    bool gridDefined = false;

    int lineNumber = 0;
    while (!feof(fp))
    {
        float x, y, dislocation, length, width, strike, dip, slip, depth;

        char line[BUFSIZ];

        if (fgets(line, BUFSIZ, fp) == NULL)
        {
            break;
        }

        ++lineNumber;

        // Ignore comments or blank lines
        if (strlen(line) <= 1 || line[0] == '#')
        {
            continue;
        }

        if (!gridDefined)
        {
            // Read grid parameters
            if (sscanf(line, "%f%f%d%d%f%f", &x0ll, &y0ll, &rows, &columns, &dx, &dy) != 6)
            {
                cerr << "Error! first readable line must contain grid parameters:" << endl
                     << "x0 y0 dislocation length width dip strike rake depth" << endl;
                fclose(fp);
                return {status::FAILURE, {}, {}, {}, {}, {}, {}, {NULL}, {}};
            }
            gridDefined = true;
        }
        else
        {
            // Read fault parameters
            if (sscanf(line, "%f%f%f%f%f%f%f%f%f",
                       &x,
                       &y,
                       &dislocation,
                       &length,
                       &width,
                       &strike,
                       &dip,
                       &slip,
                       &depth) != 9)
            {
                cerr << "Invalid parameters: line " << lineNumber << " on " << path << endl;
                continue;
            }
            faults.push_back({x, y, dislocation, length, width, strike, dip, slip, depth});
        }
    }

    fclose(fp);

    nCompoments = faults.size();
    if (gridDefined && nCompoments > 0)
    {
        faultComponents = (fault *)malloc(nCompoments * sizeof(fault));
        std::copy(faults.begin(), faults.end(), faultComponents);
        return {status::SUCCESS, x0ll, y0ll, rows, columns, dx, dy, faultComponents, nCompoments};
    }

    return {status::FAILURE, {}, {}, {}, {}, {}, {}, {NULL}, {}};
}

/**
 * @brief Calculates the deformation on CPU
 *
 * @param path Path of the configuration file
 * @param createGrids true to create output grids
 * @param elapsedTime Reference to store spent running the model
 * @return status SUCCESS when successful, ERROR when error has occured.
 */
status deformCpu(const char *path, bool createGrids, float &elapsedTime);

/**
 * @brief Calculates the deformation on GPU
 *
 * @param path Path of the configuration file
 * @param createGrids true to create output grids
 * @param elapsedTime Reference to store spent running the model
 * @return status SUCCESS when successful, ERROR when error has occured.
 */
status deformGpu(const char *path, bool createGrids, float &elapsedTime);

/**
 * @brief Prints program usage.
 *
 * @param program
 */
void usage(const char *program);

/**
 * @brief Main program.
 *
 * @param argc Command line argument count
 * @param argv Command line argument values
 * @return int Exit status, EXIT_SUCCESS when successful, EXIT_FAILURE otherwise.
 */
int main(int argc, char *argv[])
{


    char *path = NULL;

    if (argc > 1){
        path = argv[1];
    }

    if (path == NULL)
    {
        usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    bool createGrids = true;

    if (argc == 3 && string(argv[2]).compare("false") == 0)
    {
        createGrids = false;
    }

    if (!fs::exists(path) || !fs::is_regular_file(path))
    {
        cerr << path << " int a valid file or it is not readable" << endl;
        exit(EXIT_FAILURE);
    }

    // Check for environment settings

    string envGPU = Env::get("gpu");
    string envCPU = Env::get("cpu");

    bool calculateGPU = envGPU.size() == 0 || envGPU.compare("true") == 0 || envGPU.compare("TRUE") == 0;
    bool calculateCPU = envCPU.size() == 0 || envCPU.compare("true") == 0 || envCPU.compare("TRUE") == 0;

    float gpuTime{0.0f};
    float cpuTime{0.0f};

    // Calculate deformation on GPU
    if (calculateGPU) {
        deformGpu(path, createGrids, cpuTime);
    }

    // Calculate deformation on CPU
    if (calculateCPU) {
        deformCpu(path, createGrids, gpuTime);
    }

    if (gpuTime > 0.0f && cpuTime > 0.0f)
    {
        float speedUp = gpuTime / cpuTime;
        cout << "GPU speed up: " << std::fixed << std::setprecision(2) << speedUp << "x" << endl;
    }
}

void usage(const char *program)
{
    cerr << "Usage: " << program << " SCENARIO.txt [true | false]" << endl
         << endl
         << " Loads the configuration from SCENARIO.txt and outputs " << endl
         << " Ux, Uy and Uz to SCENARIO_x.asc, SCENARIO_y.asc and SCENARIO_z.asc" << endl
         << " Set the second command line argument to argument to false if you don't want the output grids to be created." << endl
         << " Example: " << endl
         << "  # Simulates scenario from samples/scenario1.txt and writes output grids to samples/scenario1_x.asc, ..." << endl
         << "  " << program << " samples/scenario1.txt" << endl
         << endl
         << "  # Simulates scenario from samples/scenario1.txt but does not create output grids:" << endl
         << "  " << program << " samples/scenario1.txt false" << endl;
}

status deformCpu(const char *path, bool createGrids, float &elapsedTime)
{
    Timer t;

    okada85::parameters params;

    const fs::path p(path);

    string filenameWithoutExt = p.stem().string();

    fs::path uzPath(p);
    fs::path uxPath(p);
    fs::path uyPath(p);

    uzPath.replace_filename(filenameWithoutExt + "_cpu_z.asc");
    uxPath.replace_filename(filenameWithoutExt + "_cpu_x.asc");
    uyPath.replace_filename(filenameWithoutExt + "_cpu_y.asc");

    // Load scenario
    auto [status, x0ll, y0ll, rows, columns, dx, dy, components, nComponents] = loadScenario(path);

    if (status == status::FAILURE)
    {
        cerr << "Unable to load configuration " << path << endl;
        exit(EXIT_FAILURE);
    }

    int numElements = rows * columns;

    auto [hostMemoryStatus, Uz, Us, Ud, Ux, Uy] = allocateHostGrids(numElements);

    if (hostMemoryStatus != status::SUCCESS)
    {
        return status::FAILURE;
    }

    cout << "Grid parameters" << endl
         << "  Origin (lower left coordinates): " << x0ll << "," << y0ll << endl
         << "  Rows: " << rows << " columns: " << columns << " Total points: " << numElements << endl
         << "  Resolution x,y (meters): " << dx << "," << dy << endl
         << "  Total of fault components: " << nComponents << endl;

    cout << "Starting CPU simulation..." << endl;

    // Mark start timestamp
    t.mark("start");

    // Run the simulation on CPU
    okada85cpu::deform(
        rows,
        columns,
        x0ll,
        y0ll,
        dx,
        dy,
        components,
        nComponents,
        params,
        Uz, Us, Ud, Ux, Uy);

    // Get elapsed seconds from start timestamp before any output is performed
    elapsedTime = t.seconds("start");

    cout << "CPU simulation finished." << endl;
    cout << "CPU time: " << elapsedTime << " seconds" << endl;

    // Save the results if needed
    if (createGrids)
    {
        cout << "Saving cpu results..." << endl;

        // Calculate cellsize in degrees
        auto [dxDeg, dyDeg] = geo::cellSizeDegrees(y0ll, dx, dy);

        // Save results to ASCII grid files
        geo::Esri::saveAscii(uzPath.string().c_str(), Uz, rows, columns, x0ll, y0ll, dxDeg, dyDeg);
        cout << "  z: " << uzPath.string() << endl;
        geo::Esri::saveAscii(uxPath.string().c_str(), Ux, rows, columns, x0ll, y0ll, dxDeg, dyDeg);
        cout << "  x: " << uxPath.string() << endl;
        geo::Esri::saveAscii(uyPath.string().c_str(), Uy, rows, columns, x0ll, y0ll, dxDeg, dyDeg);
        cout << "  y: " << uyPath.string() << endl;
    }

    releaseHostMemory({Uz, Us, Ud, Ux, Uy});

    return status::SUCCESS;
}

status deformGpu(const char *path, bool createGrids, float &elapsedTime)
{
    Timer t;

    cudaError_t cudaStatus = detectCuda();

    if (cudaStatus != cudaSuccess)
    {
        cerr << cudaGetErrorString(cudaStatus) << endl;
        return status::FAILURE;
    }

    okada85::parameters params;

    const fs::path p(path);

    string filenameWithoutExt = p.stem().string();

    fs::path uzPath(p);
    fs::path uxPath(p);
    fs::path uyPath(p);

    uzPath.replace_filename(filenameWithoutExt + "_gpu_z.asc");
    uxPath.replace_filename(filenameWithoutExt + "_gpu_x.asc");
    uyPath.replace_filename(filenameWithoutExt + "_gpu_y.asc");

    // Load scenario
    auto [status, x0ll, y0ll, rows, columns, dx, dy, components, nComponents] = loadScenario(path);

    if (status == status::FAILURE)
    {
        cerr << "Unable to load configuration " << path << endl;
        exit(EXIT_FAILURE);
    }

    int numElements = rows * columns;

    auto [hostMemoryStatus, Uz, Us, Ud, Ux, Uy] = allocateHostGrids(numElements);

    if (hostMemoryStatus != status::SUCCESS)
    {
        return status::FAILURE;
    }

    auto [deviceMemoryStatus, d_Uz, d_Us, d_Ud, d_Ux, d_Uy] = allocateDeviceGrids(Uz, Us, Ud, Ux, Uy, numElements);
    if (deviceMemoryStatus != status::SUCCESS)
    {
        return status::FAILURE;
    }

    cout << "Grid parameters" << endl
         << "  Origin (lower left coordinates): " << x0ll << "," << y0ll << endl
         << "  Rows: " << rows << " columns: " << columns << " Total points: " << numElements << endl
         << "  Resolution x,y (meters): " << dx << "," << dy << endl
         << "  Total of fault components: " << nComponents << endl;

    cout << "Starting GPU simulation..." << endl;

    // Mark start timestamp
    t.mark("start");

    // Run the simulation on GPU
    okadaStatus deformResult = okada85gpu::deform(
        rows,
        columns,
        x0ll,
        y0ll,
        dx,
        dy,
        components,
        nComponents,
        params,
        d_Uz, d_Us, d_Ud, d_Ux, d_Uy);

    // Get elapsed seconds from start timestamp before any output is performed
    elapsedTime = t.seconds("start");

    if (deformResult != okadaStatus::SUCCESS)
    {
        cerr << "GPU deformation failed." << endl;
        releaseHostMemory({Uz, Us, Ud, Ux, Uy});
        releaseDeviceMemory({d_Uz, d_Us, d_Ud, d_Ux, d_Uy});
        return status::FAILURE;
    }

    cout << "GPU simulation finished." << endl;
    cout << "GPU time: " << elapsedTime << " seconds" << endl;

    // Save the results if needed
    if (createGrids)
    {
        cout << "Saving GPU results..." << endl;

        int n = rows * columns;
        cudaError_t cudaStatus;

        cudaStatus = copyToHost(Uz, d_Uz, n);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Unable to copy device Uz to host" << cudaGetErrorString(cudaStatus) << endl;
        }

        cudaStatus = copyToHost(Ux, d_Ux, n);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Unable to copy device Ux to host" << cudaGetErrorString(cudaStatus) << endl;
        }

        cudaStatus = copyToHost(Uy, d_Uy, n);
        if (cudaStatus != cudaSuccess)
        {
            cerr << "Unable to copy device Uy to host" << cudaGetErrorString(cudaStatus) << endl;
        }

        // Calculate cellsize in degrees
        auto [dxDeg, dyDeg] = geo::cellSizeDegrees(y0ll, dx, dy);

        // Save results to ASCII grid files
        geo::Esri::saveAscii(uzPath.string().c_str(), Uz, rows, columns, x0ll, y0ll, dxDeg, dyDeg);
        cout << "  z: " << uzPath.string() << endl;
        geo::Esri::saveAscii(uxPath.string().c_str(), Ux, rows, columns, x0ll, y0ll, dxDeg, dyDeg);
        cout << "  x: " << uxPath.string() << endl;
        geo::Esri::saveAscii(uyPath.string().c_str(), Uy, rows, columns, x0ll, y0ll, dxDeg, dyDeg);
        cout << "  y: " << uyPath.string() << endl;
    }

    releaseHostMemory({Uz, Us, Ud, Ux, Uy});
    releaseDeviceMemory({d_Uz, d_Us, d_Ud, d_Ux, d_Uy});

    return status::SUCCESS;
}