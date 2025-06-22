# CPU/GPU Implementation of the Okada deformation model due to a finite rectangular earthquake source

Please review the [Project Presentation](./project_presentation/project_presentation.pdf) for project details.

According to Wikipedia, faults are fractures in the Earth's crust where rocks have moved relative to each other,
often occurring at plate boundaries [1]. The rapid movement of these faults causes earthquakes, and in the case
of undersea faults, Tsunami waves can be formed.

In order to study the source and effects of earthquakes (and tsunamis), several mathematical and computational
models have been developed, being Okada's 1985 (and 1992 revision) [2] one of the most widely used.

The deformation of the earth surface due to a finite rectangular earthquake source is a complex physical problem
that can be expressed as a set of equations to be evaluated on every one of the points of a rectangular grid,
representing a portion of the Earth surface.

## TL;DR (Summary)

- Comparison between CPU version and GPU version of Okada model over a region (grid) given one or more
  fault events.
- Sample input is provided on the samples/ folder.
- No input bathymetry is required. However, the resulting grids are correctly located using WGS84
  and the obtained results can be used to deform the bathymetry on the same region of the grids.
- NOTES:
  - Default execution stores the resulting grids on disk, see Execution and data input section for details.
  - Execution time of the CPU time is orders of magnitude larger than the GPU time for huge grids or several fault events.
  - Resulting grids can use a lot of disk space. Set the parameters with caution.

## Compiling and running with CMake (Windows - Linux)

After cloning this repository:

- Ensure CUDA Toolkit, Visual Studio Community and Visual Studio Code are installed and configured.
- Visual Studio Code plugins required: C/C++ Extension Pack, CMake Tools, VS Code Action Buttons
- On Windows, Visual Studio which contains the cl.exe compiler is required.
- Clone this repository
- Open the project folder in VS Code, wait for CMake configuration to be performed automatically.
- Some prompts are presented, configure the build to amd64 (64 bits) to ensurwe compatibility with the installed CUDA libraries.
- After the cmake configuration is done, run the executable and supply a configuration file.
- Visualize the results using a text editor or GIS software.

## Compiling and runing (Linux)

After cloning this repository:

- Same instructions, Visual Studio Community is not required, g++ is used instead.
- Ensure Development Tools meta-package (found on most Linux distributions, contains compiler, linker, etc.) is installed.
- Use the action buttons to clean, build and run the sample scenarios.
- Actions for the sample scenarios run clean and build first, so a fresh build is always used.

## Compiling and running via command line (Linux)

After cloning this repository:

- Ensure CUDA Toolkit and development tools are installed and configured.
- make clean build
- make run (runs the program on samples/test.txt)

## Motivation

This code was developed as the Capstone Project of the [GPU Programming Specialization from Coursera](https://www.coursera.org/specializations/gpu-programming)
by Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com> as a means to explore and apply the capabilities of CUDA and NVIDIA GPUs.

## Results

- Successfully developed a CPU and GPU version of the Okada model
- The GPU version reduces the calculation time in orders of magnitude.
- To ensure reproducibility of the results, an explanation of the input, output and test scenarios were provided.

## Challenges addressed on this project

- Individual calculation on each point of the grid: Several equations need to be evaluated on each one of the points inside the grid,
  taking care of special cases and singularities (division by zero, logarithms, and some others).
  Scalar-to-matrix or matrix-to-matrix operations can't be applied, because each point on the grid has different conditions
  to be evaluated on the equations and lead to different execution paths.
- Calculation over huge grids: This is a direct consequence of the prior statement.  The amount of points the equations
  are evaluated is proportional to the number of points of the grid. If the study area (grid) covers hundreds or thousands of kilometers,
  and the grid resolution is small, millions of points need to be evaluated individually.
  For instance, a rectangular grid covering from Tonga (Polynesia) to the Pacific coast of South America
  (take lower left lat,lon -178.416666,  -55.81250 to upper left lat,lon -68.5541666 16.9624999) consists of
  26367 columns and 17466 rows. With a 15-arc second separation between points, a total of 460.526.022 individual points must be evaluated.
- Multiple faults: The Okada model can also be applied to multiple overlapping small faults. The final deformation is the sum
  of the deformation caused by each one of the small faults, but the effect of each fault must be calculated invidually
  over the whole grid. If a thousand small faults occur, the calculation of each fault must be performed individually
  and then the next fault, until all faults have been evaluated.

The developed code addresses all of the three challenges. For the second one, the current version is
limited only by the amount of device and host memory on the system.

NOTE: you can test the limits of your own system by testing the execution of the samples/huge_grid_1.txt file. On a common PC
or laptop, this scenario should fail, due to the inmense amount of device memory required to store the temporary grids used
by the program. If your system runs this scenario and does not fail, congrats! You're lucky!.
Each ouput grid will take approximately 4.5 GB of disk space.

## Proposed solution

In order to test the effect of using CUDA to paralellize this complex computation, two implementations were developed:
a CPU-only and GPU accelerated version. Both are newly developed code, after performing a direct analysis,
interpretation and coding based on the model proposed on Okada's 1985 paper and also after reviewing other implementations for reference.

Some test scenarios were proposed, and new ones can be configured to test the results.

The main program allows the user to select one of several sample scenarios, runs the deformation model on CPU and GPU,
compares and prints execution times for each case.

Feel free to run all the simulation scenarios. Please remember: CPU implementations can take a large amount of
time. Programs can be interrupted at any time via keyboard interrupt. It's highly possible that the samples/huge_grid_1.txt
scenario fails, due to the dimensions and resolution of the resulting grids.

## Execution and data input

A lot of parameters are required for each simulation scenario, so files are used to store the scenario configuration. The program receives two input parameters: path to the configuration file and flag to indicate if output grids are to be written to disk.

The path to a configuration file must de provided via the command line. It contains all the parameters required for a simulation.
A second parameter set to false (default is true) allows the user to define if output grids have to be written to disk or not.

Once the program is compiled,

```sh
# Clean older build and older resulting grids
make clean build

# Run the simulation using the configuration defined on samples/averroes.txt
# Generates samples/averroes_x.txt samples/averroes_y.txt samples/averroes_z.txt
# Prints the comparison of the execution times and GPU speed up.
./okada_deform samples/averroes.txt

# Same as before, generates x y and z grids
# Prints the comparison of the execution times and GPU speed up.
./okada_deform samples/chile_1960.txt

# Same as before, but no grids are saved to disk.
# Prints the comparison of the execution times and GPU speed up.
./okada_deform samples/chile_1960.txt false

```

For simplicity on the main program, all parameters on the configuration file are assumed to be valid, minor checks are performed. The required structure is explained below.

The file must contain at least two lines. The first line defines the location, dimensions and resolution of the grid. The the following lines (one or more)
define fault event parameters. One single line after the grid definition means a single fault event, multiple lines are used to simulate
multiple fault scenarios on the same grid.

```txt
x0ll y0ll rows columns dx dy
x0 y0 dislocation length width dip strike rake depth
...
x0 y0 dislocation length width dip strike rake depth
```

Faults planes (rectangles) should be located inside or near the grid area for the effects to be noticeable.

The parameters on the first line define the grid location, boundaries and resolution:

- x0ll, y0ll: Coordinates of the lower left corner of the grid, in decimal degrees. x0ll is longitude (+East, -West), y0ll is latitude (+North, -South).
- rows, columns: Number of grid points in the x (columns - longitude) and y (rows - latitude) direction.
- dx, dy:  Grid resolution (spacing between points inside the grid) in meters.

The parameters defining the rectangular fault planes are:

- x0, y0: Origin of the fault plane (decimal degrees).
- dislocation: Displacement of the fault plane in the z direction (meters)
- length: Fault plane length on direction of the strike angle (meters)
- width: Fault plane width perpendicular to the strike angle (meters)
- dip, strike, rake: Angles of the fault plane (degrees).
- depth: Depth of the event from the earth surface (meters).

The samples/ folder contains some input configuration examples.

## Program output

The program calculates and stores the results on several files, using the scenario filename as a template.

For an input configuration called "test.txt", the program stores the results on several ESRI ASCII grids (.asc) described as follows:

- Input: test.txt
- test_z.asc: Deformation on the Z axis (up-down)
- test_x.asc: Deformation on the X axis (East-West)
- test_y.asc: Deformation on the Y axis (North-South)

Some other intermediate results are calculated by the model but they aren't saved into files.

ESRI ASCII grid files can be opened and visualized by several GIS programs such as:

- QGIS [3] (free). Recommended.
- Surfer [4] (propietary).
- ArcMap/ArcGIS [5] (propietary).
- and many more program/utilities to transform/change format such as gdal_translate, gmt, etc.

## Sample data

The samples/folder contains the some configuration scenarios. Please bear in mind that fault data and grid configurations
are approximate values. A literature review must be performed to get the accurate simulation parameters.

- averroes.txt: Single fault plane for an event on the Averroes fault at the Alboran Sea, near Spain. Related info can be found in [6].
- chile_1960.txt: Single fault plane similar to the event that caused the 1960 earthquake in Chile. The source of this event is described in [7].
- ecuador.txt: Single fault plane for an event near event that caused the of Ecuador.
- tonga.txt: Single fault plane for an event near the island of Tonga, South Pacific.
- huge_grid_1.txt: THIS SCENARIO MIGT FAIL on CUDA devices with less than 18 GB memory on the GPU! You have been warned.

Some other configurations are provided. Feel free to create copies of the files and modify the grid and fault plane parameters.
QGIS is recommended to visualize the resulting grids.

For some hints about the location of the faults, see the image of the
[Tectonic Plates of the Earth](https://www.usgs.gov/media/images/tectonic-plates-earth). Usually the faults are parallel to these lines.

## Creating fault planes

Fault planes (rectangles) can be created from visual approximation of published images on research papers.

Note: Dip angle is not visible, it's measured as the angle the plane is tilted on the Z axis (up-down).

GIS can be used to create approximate fault planes:

1. First install the QuickMapServices extension (Plugins -> Manage and install plugins).
2. From the Web->QuickMapServices menu, select "Settings".
3. Go to the "More services" tab, click "Get contributed pack". Close dialog.
4. Go again to Web->QuickMapServices, select "Google->Google Satellite" ad "USGS->Tectonic plates"
5. Create a new vector layer (Shapefile), polygon, EPGS: 4326 WGS 84.
6. From the toolbar: Toggle editing on, Add Polygon (Ctrl .)
7. Draw and approximate rectangle, with one of its sides (usually the "length"side) parallel to a tectonic plate boundary. This defines the strike angle.
8. Alternatively use [Destination point given distance and bearing from start point](https://www.movable-type.co.uk/scripts/latlong.html) to get the coordinates of the four points from x0, y0 using the desired length and width. Click the "View map" link on that section to see the two points in the map. "Bearing" is the strike angle, measured clockwise from the North, parallel to the base of the rectangle.
9. Dip, rake, dislocation and depth can be changed to correspond with literature data (or by trial and error).

## Disclaimer

Although a considerable amount of time was used to study the model, develop, refactor and organize the code, some bugs may exist.
Please review this code carefully if you intend to use it as part of your computations. Feel free to contact me and share
your thoughts / changes to improve the original code, and please give credit where is due.

## Copyright

Author: 2025 Erwin Meza Vega <emezav@unicauca.edu.co> <emezav@gmail.com>

## Third party packages

This project uses the [Geo single header library](https://github.com/emezav/geo) to save the resulting grids in ESRI ASCII format.

MIT License

## References

  1. [Fault (Geology)](https://en.wikipedia.org/wiki/Fault_(geology))
  2. [Okada, Yoshimitsu. Surface deformation due to shear and tensile faults in a half-space](https://doi.org/10.1785/BSSA0750041135), public version available on [National Research Institute for Earth Science and Disaster Resilience](https://www.bosai.go.jp/e/sp/pdf/Okada_1985_BSSA.pdf)
  3. [QGis - Spatial without compromise](https://qgis.org/)
  4. [Golden Software - Surfer](https://www.goldensoftware.com/products/surfer/)
  5. [Arcmap | ArcGIS Desktop](https://desktop.arcgis.com/es/desktop/index.html)
  6. [The Averroes Fault: a main tsunamigenic structure in the westernmost Mediterranean](https://digital.csic.es/bitstream/10261/193345/1/Gonzalez_Vida_et_al_2018_poster.pdf)
  7. [Source Estimate for the 1960 Chile Earthquake From Joint Inversion of Geodetic and Transoceanic Tsunami Data](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JB016996)
