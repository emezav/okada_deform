#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <vector>

// CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/**
 * @brief Detects CUDA capable devices
 * @return cudaSuccess if there is at least one available device, cudaErrorDevicesUnavailable if not.
 */
inline cudaError_t detectCuda()
{
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess)
    {
        return cudaStatus;
    }

    if (deviceCount == 0)
    {
        return cudaError_t::cudaErrorDevicesUnavailable;
    }

    return cudaSuccess;
}


/**
 * @brief Allocates and sets host memory to zero
 * @param h_ptr Reference to the host memory pointer
 * @param n Count of elements of size T
 * @return cudaSuccess if memory is allocated, error otherwise.
 * @note Setting memory to zero can take some time.
 */
template <typename T>
cudaError_t allocateAndZeroHostMemory(T **h_ptr, size_t n)
{
    // Allocate and zero host memory
    void * ptr = calloc(n, sizeof(T));
    if (ptr != NULL) {
        *h_ptr = ptr;
        return cudaSuccess;
    }
    return cudaError::cudaErrorMemoryAllocation;
}

/**
 * @brief Allocates host memory
 * @param h_ptr Reference to the host memory pointer
 * @param n Count of elements of size T
 * @return cudaSuccess if memory is allocated, error otherwise.
 */
template <typename T>
cudaError_t allocateHostMemory(T **h_ptr, size_t n)
{
    // Allocate host memory
    void * ptr = malloc(n * sizeof(T));
    if (ptr != NULL) {
        *h_ptr = ptr;
        return cudaSuccess;
    }
    return cudaError::cudaErrorMemoryAllocation;
}

/**
 * @brief Allocates device memory
 * @param d_ptr Reference to the device memory pointer
 * @param n Count of elements of size T
 * @return cudaSuccess if memory is allocated, error otherwise.
 */
template <typename T>
cudaError_t allocateDeviceMemory(T **d_ptr, size_t n)
{
    // Allocate device memory and return result status
    return cudaMalloc(d_ptr, n * sizeof(T));
}

/**
 * @brief Allocates device memory and copies host data to the newly allocated device memory
 * @param d_ptr Reference to the device memory pointer
 * @param h_ptr Host pointer with data
 * @param n Count of elements of size T
 * @return cudaSuccess if allocation and copy to device succeeded, error otherwise.
 */
template <typename T>
cudaError_t allocateAndCopyToDevice(T **d_ptr, T *h_ptr, size_t n)
{
    // Allocate memory and check for errors.
    cudaError_t cudaStatus = cudaMalloc(d_ptr, n * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        return cudaStatus;
    }

    // Copy data and return result status
    return cudaMemcpy(*d_ptr, h_ptr,
                      n * sizeof(T), cudaMemcpyHostToDevice);
}

/**
 * @brief Copies device data to host memory
 * @param h_ptr Pointer to host memory
 * @param d_ptr Pointer to device memory
 * @param n Count of elements of size T
 * @return cudaSuccess if copy is successful, error otherwise.
 */
template <typename T>
cudaError_t copyToHost(T *h_ptr, T *d_ptr, size_t n)
{
    return cudaMemcpy(h_ptr, d_ptr,
                      n * sizeof(T), cudaMemcpyDeviceToHost);
}

/**
 * @brief Copies data into device memory
 * @param d_ptr Pointer to device memory
 * @param h_ptr Pointer to host memory
 * @param n Count of elements of size T
 */
template <typename T>
cudaError_t copyToDevice(T *h_ptr, T *d_ptr, size_t n)
{
    // Copy data and return result status
    return cudaMemcpy(*d_ptr, h_ptr,
                      n * sizeof(T), cudaMemcpyHostToDevice);
}

/**
 * @brief Releases host memory for the supplied pointers
 * @param  pointers Vector of T* pointers to free
 */
template <class T>
void releaseHostMemory(std::vector<T *> pointers)
{
  for (auto ptr : pointers)
  {
    free(ptr);
  }
}

/**
 * @brief Releases device memory for the supplied pointers
 * @param  pointers Vector of T* pointers to free
 */
template <class T>
void releaseDeviceMemory(std::vector<T *> pointers)
{
  for (auto ptr : pointers)
  {
    cudaFree(ptr);
  }
}

#endif