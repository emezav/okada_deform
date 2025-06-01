/**
 * @file
 * @brief Grid utility functions
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @author 
 * @copyright MIT License
 */
#ifndef GRID_H
#define GRID_H
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <ctime>
#include <iostream>
#include <fstream>
#include <tuple>

#include "globals.h"

/**
 * @brief Grids namespace
 */
namespace grid
{
  using std::tuple;

  /**
   * @brief 2D grid
   */
  struct Grid
  {
    float *data{NULL}; /*!< Flat array of data (row1row2row3...), no padding between rows*/
    int rows{0};       /*!< Grid rows */
    int columns{0};    /*!< Grid columns */
    float x0{0};       /*!< X coordinate (longitude, decimal degrees) of the lower left corner */
    float y0{0};       /*!< Y coordinate (latitude, decimal degrees) of the lower left corner */
    float dx;          /*!< X resolution in meters */
    float dy;          /*!< Y resolution in meters */
    float dxdeg;       /*!< X resolution in decimal degrees */
    float dydeg;       /*!< Y resolution in decimal degrees */
  };

  /**
   * @brief Saves a 2D grid into an ESRI ASCII file
   * ESRI ASCII is a format understood by several GIS packages such as GDAL, QGIS, ArcGis, Surfer, etc.
   * @param path to save the file
   * @param g Grid to save (flattened 2D array)
   * @param rows Grid rows
   * @param columns Grid columns
   * @param x0ll Lower left corner x coordinate (longitude), decimal degrees
   * @param y0ll Lower left corder y coordinate (latitude), decimal degrees
   * @param dx Grid X resolution (meters)
   * @param dy Grid Y resolution (meters)
   * @param nodata value to be considered as NODATA
   */
  status saveAsciiGrid(const char *path, float *g, int rows, int columns, float x0ll, float y0ll, float dx, float dy, float nodata = NAN);

  /**
   * @brief Reverses the order of the rows
   * @param g Grid (rows x columns)
   * @param rows Grid rows
   * @param columns Grid columns
   */
  void flipud(float *g, int rows, int columns);

  /**
   * @brief Returns a reference to an element inside a 2D array, Row Major
   * @param m 2D array
   * @param row Row
   * @param column Column
   * @param columns Total of columns
   * @return Reference to the element.
   */
  inline float &at(float *m, int row, int column, int columns)
  {
    return m[((row)*columns) + column];
  }

  /**
   * @brief Returns the linear position for an element inside a 2D array, Row Major
   * @param row Row
   * @param column Column
   * @param columns Total of columns
   * @return Linear position
   */
  inline int linear2D(int row, int column, int columns)
  {
    return (row * columns) + column;
  }

}
#endif