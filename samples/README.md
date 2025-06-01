# Sample scenario configurations

This folder contains some configuration scenarios. Please bear in mind that fault data and grid configurations
are approximate values. A literature review must be performed to get the accurate simulation parameters.

- test.txt: Copy of ecuador.txt.
- averroes.txt: Single fault plane for an event on the Averroes fault at the Alboran Sea, near Spain. Related info can be found in [6].
- chile_1960.txt: Single fault plane similar to the event that caused the 1960 earthquake in Chile. The source of this event is described in [7].
- ecuador.txt: Single fault plane for an event near event that caused the of Ecuador.
- tonga.txt: Single fault plane for an event near the island of Tonga, South Pacific.
- huge_grid_1.txt: THIS SCENARIO MIGT FAIL on CUDA devices with less than 18 GB memory on the GPU! You have been warned.

Some other configurations are provided. Feel free to create copies of the files and modify the grid and fault plane parameters.
QGIS is recommended to visualize the resulting grids.

For some hints about the location of the faults, see the image of the
[Tectonic Plates of the Earth](https://www.usgs.gov/media/images/tectonic-plates-earth). Usually the faults are parallel to these lines.
