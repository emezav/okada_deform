/**
 * @file
 * @brief Grid utility functions
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 */

#include <fstream>
#include <iostream>
#include <iomanip>

#include "globals.h"
#include "grid.h"

namespace grid
{
    using std::cerr;
    using std::cout;
    using std::endl;
    using std::ofstream;

    status saveAsciiGrid(const char *path, float *g, int rows, int columns, float x0ll, float y0ll, float dx, float dy, float nodata)
    {
        ofstream ofs(path);

        if (!ofs.is_open())
        {
            cerr << "Unable to open " << path << endl;
            return status::FAILURE;
        }

        // Calculate dx,dy in decimal degrees from dx,dy in meters at the latitude of the grid origin
        auto [dxDeg, dyDeg] = cellSizeDegrees(y0ll, dx, dy);

        // Write ASCII grid header
        ofs << "ncols " << columns << endl;
        ofs << "nrows " << rows << endl;
        ofs << "xllcorner " << std::fixed << std::setprecision(6) << x0ll << endl;
        ofs << "yllcorner " << std::fixed << std::setprecision(6) << y0ll << endl;
        ofs << "dx " << std::fixed << std::setprecision(7) << dxDeg << endl;
        ofs << "dy " << std::fixed << std::setprecision(7) << dyDeg << endl;
        ofs << "NODATA_value " << nodata << endl;

        // Write data (last row first!)
        for (int j = rows - 1; j >= 0; j--)
        {
            for (int i = 0; i < columns; i++)
            {
                int pos = (j * columns) + i;
                ofs << ((i > 0) ? " " : "") << g[pos];
            }
            ofs << endl;
        }
        return status::SUCCESS;
    }

    void flipud(float *g, int rows, int columns)
    {
        for (int i = 0; i < rows / 2; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                int pos = (i * columns) + j;
                int newPos = ((rows - i - 1) * columns) + j;
                std::swap(g[pos], g[newPos]);
            }
        }
    }

    /**
     * @brief Fills a grid with a value
     *
     * @param g Grid to fill
     * @param rows Number of rows on the grid
     * @param columns Number of columns on the grid
     * @param value Value to fill on each grid cell
     */
    void fill(float *g, int rows, int columns, float value)
    {
        for (int i = 0; i < rows * columns; i++)
        {
            g[i] = value;
        }
    }

}
