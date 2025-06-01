# Create Point Shapefile from points

On QGIS, go to Layer -> Create Layer -> New virtual layer...

Enter the query with the coordinates of the fout points of the rectangle.

For instance, for the Ecuador fault:

```sql
select make_polygon(make_line(
    make_point(-81.5, 2.7, 4326),
    make_point(-79.630278, 1.619444, 4326),
    make_point(-81.878611, -2.275, 4326),
    make_point(-83.746944, -1.195278, 4326),
    make_point(-81.5, 2.7, 4326)
)) as geometry
```

Note that the first corner of the rectangle corresponds to the x0, y0 coordinates on samples/ecuador.txt

- Second corner: 2 * width (240 kilometers) perpendicular to the strike (30 + 90 = 120 degrees bearing from first corner)
- Third corner: length (500 kilometers) south from second corner, bearing parallel to the strike (180 + 30 = 210 degrees) 
- Fourth corner: 2 * width perpendicular to the strike (120 degrees bearing) negative distance from the third corner, or 270 + strike angle (270 + 30 = 300) positive distance.
- Fifth point must be the first corner to close the polygon.
