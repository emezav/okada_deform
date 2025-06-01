/**
 * @file
 * @brief Fault definition on Okada deformation model
 * Okada Y., Surface deformation due to shear and tensile faults in a  half-space
 * Bull. Seismol. Soc. Am., 75:4, 1135-1154, 1985.
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 */

#ifndef OKADA85_H
#define OKADA85_H

#include "globals.h"

/**
 * @brief Common definitions for the Okada model.
 * 
 */
namespace okada85
{

  /**
   * @brief Fault component.
   */
  struct fault
  {
    float lon{};    /*!< Origin of the fault - longitude */
    float lat{};    /*!< Origin of the fault - latitude */
    float d{};      /*!< Dislocation (m) */
    float length{}; /*!< Height of the fault (m) */
    float width{};  /*!< Width of the fault (m) */
    float st{};     /*!< Strike (deg) */
    float di{};     /*!< Dip (deg) */
    float sl{};     /*!<Slip (deg) */
    float hh{};     /*!< Depth of the upper fault edge (m) */
  };

  /**
   * @brief Parameters relevant to this context, can be changed by the user.
   */
  struct parameters
  {
    float vp{8.1e3f};       /*!< P-Wave velocity (m/sec) */
    float vs{3.8e3f};       /*!< S-Wave velocity (m/sec) */
    float re{6377397.155f}; /*!< Radius of the equator (m) */
    float e2{0.006694470};  /*!< Eccentricity of the earth */
    float eps{1e-8f};       /*!< Epsilon (zero) to detect singularities in some terms (PP. 1148) */
  };
}
#endif