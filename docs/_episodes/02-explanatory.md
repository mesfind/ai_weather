---
title: Data Preprocessing and Visualization
teaching: 1
exercises: 0
questions:
- "How do I processing nc data?"
objectives:
- Understand xarray basics and capabilities.
- Load and manipulate data using xarray.
- Perform data processing and visualization tasks.
- Integrate xarray with other libraries.
- Optimize data processing pipelines.
- Explore best practices for data visualization.

---


# Data preprocessing


## Preprocessing with xarray

We will be exploring the xarray architecture using some sample climate data from the European Centre for Medium-Range Weather Forecasts (ECMWF). We will use their ERA-Intrim climate reanalysis project. You can download the data in netCDF format here. As is the case for many climate products, the process involves downloading large netCDF files to a local machine.

If you visit the ECMWF page you will find that you can download a large number of different climate fields. Here we have prepared tutorial examples around 4 variables. Note that we provide a full resolution (global) version as well as a subsetted version (Alaska). Choose the Alaska data if you are limited for space or processing power on your local computer. 

## Multidimensional arrays

N-dimensional arrays of numbers, like NumPy's ndarray, are fundamental in scientific computing, widely used for data representation. Geoscientists, in particular, rely on arrays to structure their data. For instance, climate scientists work with spatially and temporally varying climate variables (e.g., temperature, precipitation) on gridded datasets. Tasks include subsetting global grids for regional analysis, selecting specific time slices, and applying statistical functions to derive summary insights from these subsets.

![png](../fig/dataset-diagram.png)

The tools in this tutorial have some similarity to raster image processing tools. Both require computational engines that can manipulate large stacks of data formatted as arrays. Here we focus on tools that are optimized to handle data that have many variables spanning dimensions of time and space. See the raster tutorials for tools that are optimized for image processing of remote sensing datasets.

## Conventional Approach: Working with Unlabelled Arrays

Multidimensional array data are typically stored in custom binary formats and accessed through proprietary libraries (e.g., Fortran or C++). Users must manage file structures and write custom code to handle these files. Data subsetting involves loading the entire dataset into memory and using nested loops with conditional statements to identify specific index ranges for temporal or spatial slices. Additionally, matrix algebra techniques are employed to summarize data across spatial and temporal dimensions.

## Challenges

Working with N-dimensional arrays in this manner presents a significant challenge due to the disconnect between the data and its metadata. Users often struggle to interpret array indices without context, leading to inefficiencies and errors. Common issues arise when determining the placement of critical dimensions within the array or ensuring alignment after data manipulation tasks like resampling.

## The Network Common Data Format

The Network Common Data Form (netCDF) was developed in the early 1990s to address the complexities of handling N-dimensional arrays. NetCDF offers self-describing, platform-independent binary data formats and tools that simplify the creation, access, and sharing of scientific data stored in multidimensional arrays, complete with metadata. Initially designed by the climate science community to manage the growing size of regional climate model outputs, netCDF has evolved to merge with HDF5, enhancing data storage capabilities.

## Practical Use of NetCDF

NetCDF has become a standard for distributing N-dimensional arrays, with various scientific communities relying on its software tools for data processing and visualization. While some researchers exclusively utilize netCDF toolkits for tasks like subsetting and grouping, others leverage NetCDF primarily for data serialization. In cases where standard tools lack the required flexibility for specific research needs, users resort to custom coding methods for statistical analysis and subsetting operations.

## Handling Large Arrays

NetCDF imposes no file size limits, but processing tools are constrained by available memory when reading data for computational tasks. As multidimensional datasets expand in size due to higher resolutions and advanced sensing technologies, managing these large datasets becomes increasingly challenging for computational resources.


## xarray Architecture

To delve into the architecture of xarray, we initiate by importing the xarray library into our Python environment

~~~
import xarray as xr
~~~

The next step involves opening the data file and loading it into a Dataset object. It is crucial to note that the choice of engine for opening the dataset depends on the specific format of the netCDF file being utilized. This decision is pivotal for ensuring seamless data access and manipulation. Here is an example of opening a netCDF file and loading it into a Dataset:

~~~
dset = xr.open_dataset('../data/pr_Amon_ACCESS-CM2_historical.nc')
~~~
{: .python}

By executing the above code snippet, we establish a Dataset object named `ds` that encapsulates the data from the specified netCDF file. This Dataset serves as a central data structure within xarray, enabling efficient handling of multidimensional data arrays and facilitating various data processing and visualization tasks.


### Dataset Properties

Next we will ask xarray to display some of the parameters of the Dataset. To do this simply return the contents of the Dataset variable name

~~~
ds
~~~
{: .python}


~~~
<xarray.Dataset>
Dimensions:    (time: 60, bnds: 2, lon: 192, lat: 144)
Coordinates:
  * time       (time) datetime64[ns] 2010-01-16T12:00:00 ... 2014-12-16T12:00:00
  * lon        (lon) float64 0.9375 2.812 4.688 6.562 ... 355.3 357.2 359.1
  * lat        (lat) float64 -89.38 -88.12 -86.88 -85.62 ... 86.88 88.12 89.38
Dimensions without coordinates: bnds
Data variables:
    time_bnds  (time, bnds) datetime64[ns] ...
    lon_bnds   (lon, bnds) float64 ...
    lat_bnds   (lat, bnds) float64 ...
    pr         (time, lat, lon) float32 ...
Attributes: (12/50)
    CDI:                    Climate Data Interface version 1.9.8 (https://mpi...
    source:                 ACCESS-CM2 (2019): \naerosol: UKCA-GLOMAP-mode\na...
    institution:            CSIRO (Commonwealth Scientific and Industrial Res...
    Conventions:            CF-1.7 CMIP-6.2
    activity_id:            CMIP
    branch_method:          standard
    ...                     ...
    cmor_version:           3.4.0
    tracking_id:            hdl:21.14100/b4dd0f13-6073-4d10-b4e6-7d7a4401e37d
    license:                CMIP6 model data produced by CSIRO is licensed un...
    CDO:                    Climate Data Operators version 1.9.8 (https://mpi...
    history:                Tue Jan 12 14:50:25 2021: ncatted -O -a history,p...
    NCO:                    netCDF Operators version 4.9.2 (Homepage = http:/...
~~~
{: .output}

### Extracting DataArrays from a Dataset

We have queried the dataset details about our Datset dimensions, coordinates and attributes. Next we will look at the variable data contained within the dataset. In the graphic above, there are two variables (temperature and precipitation). As described above, xarray stores these observations as a DataArray, which is similar to a conventional array you would find in numpy or matlab.

Extracting a DataArray for processing is simple. From the Dataset metadata shown above, notice that the name of the climate variable is ‘t2m’ (2 meter air temperature). Suppose we want to extract that array for processing and store it to a new variable called temperature:

~~~
precipitation = dset['pr']
precipitation
~~~
{: .python}

Now, take a look at the contents of the `precipitation` variable. Note that the associated coordinates and attributes get carried along for the ride. Also note that we are still not reading any data into memory.

~~~
<xarray.DataArray 'pr' (time: 60, lat: 144, lon: 192)>
[1658880 values with dtype=float32]
Coordinates:
  * time     (time) datetime64[ns] 2010-01-16T12:00:00 ... 2014-12-16T12:00:00
  * lon      (lon) float64 0.9375 2.812 4.688 6.562 ... 353.4 355.3 357.2 359.1
  * lat      (lat) float64 -89.38 -88.12 -86.88 -85.62 ... 86.88 88.12 89.38
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
~~~
{: .output}


### Indexing

Indexing is used to select specific elements from xarray files. Let’s select some data from the 2-meter temperature DataArray. We know from the previous lesson that this DataArray has dimensions of time and two dimensional space (latitude and longitude).

You are probably already used to conventional ways of indexing an array. You will know in advance that the first array index is time, the second is latitude, and so on. You would then use positional indexing):

~~~
dset['pr'][0, 0, 0]
~~~
{: .python}

~~~
Out[22]: 
<xarray.DataArray 'pr' ()>
[1 values with dtype=float32]
Coordinates:
    time     datetime64[ns] 2010-01-16T12:00:00
    lon      float64 0.9375
    lat      float64 -89.38
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
~~~
{: .output}

This method of handling arrays should be familiar to anyone who has worked with arrays in MATLAB or NumPy. One challenge with this approach: it is not simple to associate an integer index position with something meaningful in our data. For example, we would have to write some function to map a specific date in the time dimension to its associated integer. Therefore, xarray lets us perform positional indexing using labels instead of integers:

~~~
dset['pr'].loc['2010-01-16T12:00:00', :, :]
~~~
{: .python}

~~~
<xarray.DataArray 'pr' (lat: 144, lon: 192)>
[27648 values with dtype=float32]
Coordinates:
    time     datetime64[ns] 2010-01-16T12:00:00
  * lon      (lon) float64 0.9375 2.812 4.688 6.562 ... 353.4 355.3 357.2 359.1
  * lat      (lat) float64 -89.38 -88.12 -86.88 -85.62 ... 86.88 88.12 89.38
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
~~~
{: .output}

This is great, but we still need to be keeping track of the fact that our index position 1 is the time dimension, position 2 is latitude, etc. So rather than looking up our dimension by position, xarray enables us to use the dimension name instead:

~~~
dset['pr'].isel(time=0, lat=0, lon=0)
~~~
{: .python}

~~~
<xarray.DataArray 'pr' ()>
[1 values with dtype=float32]
Coordinates:
    time     datetime64[ns] 2010-01-16T12:00:00
    lon      float64 0.9375
    lat      float64 -89.38
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
~~~
{: .output}

Here, the isel method refers to a selection by integer position. Finally, we can combine the benefits of both a labeled index and a named dimension as follows:

~~~
dset['pr'].sel(time='2010-01-16T12:00:00', lat=86.88, lon=89.38, method='nearest')
~~~
{: .python}

~~~
<xarray.DataArray 'pr' ()>
[1 values with dtype=float32]
Coordinates:
    time     datetime64[ns] 2010-01-16T12:00:00
    lon      float64 89.06
    lat      float64 86.88
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
~~~
{: .output}

## Plotting data in 1 dimension
Let’s start visualizing some of the data slices we’ve been working on so far. We will begin by creating a new variable for plotting a 1-dimensional time series:

~~~
lat_value = 75.0
lon_value = 89.0

lat_index = dset['lat'].sel(lat=lat_value, method='nearest').values
lon_index = dset['lon'].sel(lon=lon_value, method='nearest').values

time_series = dset['pr'].sel(time=slice('2010-01-16T12:00:00', '2014-12-16T12:00:00'), lat=lat_index, lon=lon_index)
time_series.plot()
~~~
{: .python}

![](../fig/time_series1.png)


Your plots can be customized using syntax that is very similar to Matplotlib. For example:

~~~
time_series.plot.line(color='green', marker='o')
~~~
{: .python}

![](../fig/time_series2.png)


### Plotting data in 2 dimensions

Since many xarray applications involve geospatial datasets, xarray’s plotting extends to maps in 2 dimensions. Let’s first select a 2-D subset of our data by choosing a single date and retaining all the latitude and longitude dimensions:

~~~
map_data = dset['pr'].sel(time='2010-01-16T12:00:00')
map_data.plot()
~~~
{: .python}

Note that in the above label-based lookup, we did not specify the latitude and longitude dimensions, in which case xarray assumes we want to return all elements in those dimensions.

![](../fig/map_time_series.png)

Customization can occur following standard Matplotlib syntax. Note that before we use matplotlib, we will have to import that library:

~~~
import matplotlib.pyplot as plt
map_data.plot(cmap=plt.cm.Blues)
plt.title('ECMWF global precipitation data')
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.tight_layout()
plt.show()
~~~
{: .python}

![](../fig/map_time_series_pr.png)

# Data processing and visualisation

## Arithmetic

Suppose we want to plot the difference in air precipitation between January 1 in 2010 versus 2011. We can do this by taking advantage of xarray’s labeled dimensions to simplify arithmetic operations on DataArray objects:

~~~
precipitation1 = dset['pr'].sel(time='2010-01-16T12:00:00')
precipitation2 = dset['pr'].sel(time='2011-01-16T12:00:00')
delta = precipitation1 - precipitation2
delta.plot()
~~~
{: .python}

![](../fig/map_time_series_pr_delta.png)

Note that the subtraction is automatically vectorized over all array values, as in numpy.


We can actually use either the dset['pr'] or dset.pr syntax to access the precipitation xarray.DataArray.

To calculate the precipitation climatology, we can make use of the fact that xarray DataArrays have built in functionality for averaging over their dimensions.

~~~
clim = dset['pr'].mean('time', keep_attrs=True)
print(clim)
~~~
{: .python}


~~~
<xarray.DataArray 'pr' (lat: 145, lon: 192)>
array([[2.4761673e-06, 2.4761673e-06, 2.4761673e-06, ..., 2.4761673e-06,
        2.4761673e-06, 2.4761673e-06],
       [2.2970205e-06, 2.2799834e-06, 2.2585127e-06, ..., 2.3540958e-06,
        2.3336945e-06, 2.3136067e-06],
       [2.0969844e-06, 2.0686068e-06, 2.0382870e-06, ..., 2.1673986e-06,
        2.1523117e-06, 2.1302694e-06],
       ...,
       [8.7852204e-06, 8.8236175e-06, 8.8202569e-06, ..., 8.7430153e-06,
        8.7706394e-06, 8.7947683e-06],
       [8.4821795e-06, 8.4632229e-06, 8.4983958e-06, ..., 8.4181611e-06,
        8.4009334e-06, 8.4525291e-06],
       [7.7492014e-06, 7.7492014e-06, 7.7492014e-06, ..., 7.7492014e-06,
        7.7492014e-06, 7.7492014e-06]], dtype=float32)
Coordinates:
  * lon      (lon) float64 0.0 1.875 3.75 5.625 7.5 ... 352.5 354.4 356.2 358.1
  * lat      (lat) float64 -90.0 -88.75 -87.5 -86.25 ... 86.25 87.5 88.75 90.0
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
~~~
{: .output}

so we can go ahead and multiply that array by 86400 and update the units attribute accordingly:

~~~
clim.data = clim.data * 86400
clim.attrs['units'] = 'mm/day' 

print(clim)
~~~
{: .python}

~~~
<xarray.DataArray 'pr' (lat: 145, lon: 192)>
array([[0.21394086, 0.21394086, 0.21394086, ..., 0.21394086, 0.21394086,
        0.21394086],
       [0.19846257, 0.19699057, 0.1951355 , ..., 0.20339388, 0.20163121,
        0.19989562],
       [0.18117945, 0.17872763, 0.176108  , ..., 0.18726324, 0.18595973,
        0.18405528],
       ...,
       [0.75904304, 0.76236055, 0.76207019, ..., 0.75539652, 0.75778324,
        0.75986798],
       [0.73286031, 0.73122246, 0.7342614 , ..., 0.72732912, 0.72584065,
        0.73029851],
       [0.669531  , 0.669531  , 0.669531  , ..., 0.669531  , 0.669531  ,
        0.669531  ]])
Coordinates:
  * lon      (lon) float64 0.0 1.875 3.75 5.625 7.5 ... 352.5 354.4 356.2 358.1
  * lat      (lat) float64 -90.0 -88.75 -87.5 -86.25 ... 86.25 87.5 88.75 90.0
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          mm/day
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
~~~
{: .output}

We could now go ahead and plot our climatology using matplotlib, but it would take many lines of code to extract all the latitude and longitude information and to setup all the plot characteristics. Recognising this burden, the xarray developers have built on top of matplotlib.pyplot to make the visualisation of xarray DataArrays much easier.

~~~
import cartopy.crs as ccrs

fig = plt.figure(figsize=[12,5])

ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))

clim.plot.contourf(ax=ax,
                   levels=np.arange(0, 13.5, 1.5),
                   extend='max',
                   transform=ccrs.PlateCarree(),
                   cbar_kwargs={'label': clim.units})
ax.coastlines()

plt.show()
~~~
{: .python}

![](../fig/map_time_series_pr_mean.png)

The default colorbar used by matplotlib is viridis. It used to be jet, but that was changed a couple of years ago in response to the #endtherainbow campaign.

Putting all the code together (and reversing viridis so that wet is purple and dry is yellow)…

~~~
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

accesscm2_pr_file = 'data/pr_Amon_ACCESS-ESM1-5_historical.nc'

dset = xr.open_dataset(accesscm2_pr_file)

clim = dset['pr'].mean('time', keep_attrs=True)

clim.data = clim.data * 86400
clim.attrs['units'] = 'mm/day'

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
clim.plot.contourf(ax=ax,
                   levels=np.arange(0, 13.5, 1.5),
                   extend='max',
                   transform=ccrs.PlateCarree(),
                   cbar_kwargs={'label': clim.units},
                   cmap='viridis_r')
ax.coastlines()
plt.show()
~~~
{: .python}

![](../fig/map_time_series_pr_mean2.png)


## Mathematical functions

Now, sometimes we need to apply mathematical functions to array data in our analysis. A good example is wind data, which are often distributed as orthogonal “u” and “v” wind components. To calculate the wind magnitude we need to take the square root of the sum of the squares. For this we use numpy ufunc commands that can operate on a DataArray. Let’s look at our wind datasets:


~~~
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cmocean

accesscm2_pr_file = 'data/pr_Amon_ACCESS-ESM1-5_historical.nc'

dset = xr.open_dataset(accesscm2_pr_file)

clim = dset['pr'].groupby('time.season').mean('time', keep_attrs=True)

clim.data = clim.data * 86400
clim.attrs['units'] = 'mm/day'

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
clim.sel(season='JJA').plot.contourf(ax=ax,
                                     levels=np.arange(0, 13.5, 1.5),
                                     extend='max',
                                     transform=ccrs.PlateCarree(),
                                     cbar_kwargs={'label': clim.units},
                                     cmap=cmocean.cm.haline_r)
ax.coastlines()

model = dset.attrs['source_id']
title = f'{model} precipitation climatology (JJA)'
plt.title(title)

plt.show()
~~~
{: .python}

![](../fig/map_time_series_pr_mean3.png)

If we wanted to create a similar plot for a different model and/or different month, we could cut and paste the code and edit accordingly. The problem with that (common) approach is that it increases the chances of a making a mistake. If we manually updated the season to ‘DJF’ for the clim.sel(season= command but forgot to update it when calling plt.title, for instance, we’d have a mismatch between the data and title.

The cut and paste approach is also much more time consuming. If we think of a better way to create this plot in future (e.g. we might want to add gridlines using plt.gca().gridlines()), then we have to find and update every copy and pasted instance of the code.

A better approach is to put the code in a function. The code itself then remains untouched, and we simply call the function with different input arguments.

~~~
def plot_pr_climatology(pr_file, season, gridlines=False):
    """Plot the precipitation climatology.
    
    Args:
      pr_file (str): Precipitation data file
      season (str): Season (3 letter abbreviation, e.g. JJA)
      gridlines (bool): Select whether to plot gridlines
    
    """

    dset = xr.open_dataset(pr_file)

    clim = dset['pr'].groupby('time.season').mean('time', keep_attrs=True)

    clim.data = clim.data * 86400
    clim.attrs['units'] = 'mm/day'

    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    clim.sel(season=season).plot.contourf(ax=ax,
                                          levels=np.arange(0, 13.5, 1.5),
                                          extend='max',
                                          transform=ccrs.PlateCarree(),
                                          cbar_kwargs={'label': clim.units},
                                          cmap=cmocean.cm.haline_r)
    ax.coastlines()
    if gridlines:
        plt.gca().gridlines()
    
    model = dset.attrs['source_id']
    title = f'{model} precipitation climatology ({season})'
    plt.title(title)
~~~
{: .python}

The docstring allows us to have good documentation for our function:

~~~
help(plot_pr_climatology)
~~~
{: .python}

~~~
Help on function plot_pr_climatology in module __main__:

plot_pr_climatology(pr_file, season, gridlines=False)
    Plot the precipitation climatology.
    
    Args:
      pr_file (str): Precipitation data file
      season (str): Season (3 letter abbreviation, e.g. JJA)
      gridlines (bool): Select whether to plot gridlines
~~~
{: .output}

We can now use this function to create exactly the same plot as before:

~~~
plot_pr_climatology('data/pr_Amon_ACCESS-ESM1-5_historical.nc', 'JJA')
plt.show()
~~~
{: .python}

![](../fig/map_time_series_pr_mean3.png)

Or use the optional gridlines input argument to change the default behaviour of the function (keyword arguments are usually used for options that the user will only want to change occasionally):

~~~
plot_pr_climatology('data/pr_Amon_ACCESS-ESM1-5_historical.nc','DJF', gridlines=True)
plt.show()
~~~
{: .python}

![](../fig/map_time_series_pr_mean4.png)

## Large data with dask

So far we’ve been working with small, individual data files that can be comfortably read into memory on a modern laptop. What if we wanted to process a larger dataset that consists of many files and/or much larger file sizes? For instance, let’s say the next step in our global precipitation analysis is to plot the daily maximum precipitation over the 1850-2014 period for the high resolution CNRM-CM6-1-HR model.

~~~
import glob
pr_files = glob.glob('data/pr_*.nc')
pr_files.sort()
print(pr_files)
~~~
{: .python}

~~~
['data/pr_Amon_ACCESS-CM2_historical.nc', 'data/pr_Amon_ACCESS-ESM1-5_historical.nc']
~~~
{: .output}

~~~
import xarray as xr
import glob
pr_files = glob.glob('data/ERSSTv5_*.nc')
pr_files.sort()

dset = xr.open_mfdataset(pr_files, combine = 'nested', concat_dim="time", chunks={'time': '500MB'})

print(dset)

~~~
{: .python}

~~~
<xarray.Dataset>
Dimensions:  (time: 859, lat: 89, lon: 180)
Coordinates:
  * time     (time) object 1950-06-01 00:00:00 ... 2021-12-15 00:00:00
  * lat      (lat) float64 -88.0 -86.0 -84.0 -82.0 -80.0 ... 82.0 84.0 86.0 88.0
  * lon      (lon) float64 0.0 2.0 4.0 6.0 8.0 ... 350.0 352.0 354.0 356.0 358.0
Data variables:
    sst      (time, lat, lon) float64 dask.array<chunksize=(859, 89, 180), meta=np.ndarray>
~~~
{: .output}


We can see that our dset object is an xarray.Dataset, but notice now that each variable has type dask.array with a chunksize attribute. Dask will access the data chunk-by-chunk (rather than all at once), which is fortunate because at 45GB the full size of our dataset is much larger than the available RAM on our laptop (17GB in this example). Dask can also distribute chunks across multiple cores if we ask it to (i.e. parallel processing).

So how big should our chunks be? As a general rule they need to be small enough to fit comfortably in memory (because multiple chunks can be in memory at once), but large enough to avoid the time cost associated with asking Dask to manage/schedule lots of little chunks. The Dask documentation suggests that chunk sizes between 10MB-1GB are common, so we’ve set the chunk size to 500MB in this example. Since our netCDF files are chunked by time, we’ve specified that the 500MB Dask chunks should also be along that axis. Performance would suffer dramatically if our Dask chunks weren’t aligned with our netCDF chunks.

~~~
 dset['sst'].data
~~~
{: .python}

~~~
 dask.array<open_dataset-sst, shape=(859, 89, 180), dtype=float64, chunksize=(859, 89, 180), chunktype=numpy.ndarray>
~~~
{: .output}

Now that we understand the chunking information contained in the metadata, let’s go ahead and calculate the daily maximum precipitation.

~~~
sst_max = dset['sst'].max('time', keep_attrs=True)
print(sst_max)
~~~
{: .python}

~~~
<xarray.DataArray 'sst' (lat: 89, lon: 180)>
dask.array<_nanmax_skip-aggregate, shape=(89, 180), dtype=float64, chunksize=(89, 180), chunktype=numpy.ndarray>
Coordinates:
  * lat      (lat) float64 -88.0 -86.0 -84.0 -82.0 -80.0 ... 82.0 84.0 86.0 88.0
  * lon      (lon) float64 0.0 2.0 4.0 6.0 8.0 ... 350.0 352.0 354.0 356.0 358.0
~~~
{: .output}

It seems like the calculation happened instataneously, but it’s actually just another “lazy” feature of xarray. It’s showing us what the output of the calculation would look like (i.e. a 360 by 720 array), but xarray won’t actually do the computation until the data is needed (e.g. to create a plot or write to a netCDF file).

To force xarray to do the computation we can use .compute() with the %%time Jupyter notebook command to record how long it takes:

~~~
%%time
sst_max_done = sst_max.compute()
sst_max_done
~~~
{: .python}

~~~
CPU times: user 103 ms, sys: 47.8 ms, total: 151 ms
Wall time: 236 ms
~~~
{: .output}

By processing the data chunk-by-chunk, we’ve avoided the memory error we would have generated had we tried to handle the whole dataset at once. A completion time of 3 minutes and 44 seconds isn’t too bad, but that was only using one core. We can try and speed things up by using a dask “client” to run the calculation in parallel across multiple cores:

~~~
from dask.distributed import Client
client = Client()
client
~~~
{: .python}

~~~
<Client: 'tcp://127.0.0.1:50335' processes=4 threads=8, 
memory=8.00 GiB>
Dashboard: http://127.0.0.1:8787/status 
~~~
{: .output}

(Click on the dashboard link to watch what’s happening on each core.)

![](dask_client_status.png)

~~~
%%time
sst_max_done = sst_max.compute()
~~~
{: .python}

~~~
CPU times: user 12 ms, sys: 5.18 ms, total: 17.2 ms
Wall time: 349 ms
~~~
{: .output}

By distributing the calculation across all four cores the processing time has dropped to 2 minutes and 33 seconds. It’s faster than, but not a quarter of, the original 3 minutes and 44 seconds because there’s a time cost associated with setting up and coordinating jobs across all the cores.


Now that we’ve computed the daily maximum precipitation, we can go ahead and plot it:

~~~
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean

sst_max_done.data = sst_max_done.data * 86400
sst_max_done.attrs['units'] = 'mm/day'

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
sst_max_done.plot.contourf(ax=ax,
                         levels=np.arange(0, 450, 50),
                         extend='max',
                         transform=ccrs.PlateCarree(),
                         cbar_kwargs={'label': sst_max_done.units},
                         cmap=cmocean.cm.haline_r)
ax.coastlines()

model = dset.attrs['source_id']
title = f'Daily maximum sst, 1850-2014 ({model})'
plt.title(title)

plt.show()
~~~
{: .python}




## Quick visualization 

In this section, we will learn to read the metadata and visualize the NetCDF file we just downloaded.

Make sure you have installed Python along with the additional packages required to read Climate 
data files as described in the [setup](../setup) instructions.

The file we downloaded from CDS should be in your `../data` folder; to check it out, open a Terminal (Git bash terminal on windows) and type:

~~~
ls ./data/*.nc
~~~
{: .language-bash}

For those of you who are not familiar with bash language:

the `~` symbol (a.k.a. *tilde*) is a shortcut for the home directory of a user;

`*.nc` means that we are looking for any files with a suffix `.nc` (NetCDF file).

~~~
adaptor.mars.internal-1559329510.4428957-10429-22-1005b553-e70d-4366-aa63-1424db2df740.nc
~~~
{: .output}

Then rename this file to a more friendly filename (please note that to ease further investigation, we add the date in the filename).

For instance, using `bash`:

~~~
mv ~/Downloads/adaptor.mars.internal-1559329510.4428957-10429-22-1005b553-e70d-4366-aa63-1424db2df740.nc ../data/ERA5_REANALYSIS_precipitation_200306.nc

ls ../data/*.nc
~~~
{: .bash}

~~~
ERA5_REANALYSIS_precipitation_200306.nc
~~~
{: .output}

[Start Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/getting-started/#navigator-starting-navigator), then select **Environments**:

<img src="../fig/anaconda-navigator-environment.png" width="80%"/>



Select the **esm-python-analysis** environment and either left-click on the triangle/arrow next to it and **Open with Jupyter Notebook**, or go back to the Home tab and click on Launch to open your **Jupyter Notebook** :

<img src="../fig/python3_notebook.png" width="80%"/>

### Get metadata

~~~
import xarray as xr

# the line above is necessary for getting 
# your plot embedded within the notebook
%matplotlib inline

dset = xr.open_dataset("~/Downloads/ERA5_REANALYSIS_precipitation_200306.nc")
print(dset)
~~~
{: .python}
      
Printing `dset` returns `ERA5_REANALYSIS_precipitation_200306.nc` metadata:

~~~
<xarray.Dataset>
Dimensions:    (latitude: 721, longitude: 1440, time: 1)
Coordinates:
  * longitude  (longitude) float32 0.0 0.25 0.5 0.75 ... 359.25 359.5 359.75
  * latitude   (latitude) float32 90.0 89.75 89.5 89.25 ... -89.5 -89.75 -90.0
  * time       (time) datetime64[ns] 2003-06-01
Data variables:
    tp         (time, latitude, longitude) float32 ...
Attributes:
    Conventions:  CF-1.6
    history:      2019-05-31 19:05:13 GMT by grib_to_netcdf-2.10.0: /opt/ecmw...
~~~
{: .output}

We can see that our `dset` object is an `xarray.Dataset`, which when printed shows all the metadata associated with our netCDF data file.

In this case, we are interested in the precipitation variable contained within that xarray Dataset:

~~~
print(dset['tp'])
~~~
{: .python}

~~~
<xarray.DataArray 'tp' (time: 1, latitude: 721, longitude: 1440)>
[1038240 values with dtype=float32]
Coordinates:
  * longitude  (longitude) float32 0.0 0.25 0.5 0.75 ... 359.25 359.5 359.75
  * latitude   (latitude) float32 90.0 89.75 89.5 89.25 ... -89.5 -89.75 -90.0
  * time       (time) datetime64[ns] 2003-06-01
Attributes:
    units:      m
    long_name:  Total precipitation
~~~
{: .output}

The [total precipitation](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation) is in units of "metre of water per day".

### Quick visualization

~~~
dset['tp'].plot()
~~~
{: .python}

<img src="../fig/tp_plot.png" />

We can change the [colormap](https://matplotlib.org/users/colormaps.html) and 
adjust the maximum (remember the total precipitation is in metre):

~~~
dset['tp'].plot(cmap='jet', vmax=0.02)
~~~
{: .python}

<img src="../fig/tp_plot_jet.png" />

We can see there is a *band* around the equator and areas especially in Asia and South America with a lot of rain. Let's add continents and a projection using cartopy:

~~~
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=[12,5])

# 111 means 1 row, 1 col and index 1
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

dset['tp'].plot(ax=ax, vmax=0.02, cmap='jet',
                   transform=ccrs.PlateCarree())
ax.coastlines()

plt.show()
~~~
{: .python}

<img src="../fig/tp_plot_jet_ccrs.png" />

At this stage, do not bother too much about the [projection](https://scitools.org.uk/cartopy/docs/latest/crs/projections.html) e.g. `ccrs.PlateCarree`. 
We will discuss it in-depth in a follow-up episode.


> ## Retrieve surface air temperature
> 
> From the same product type ([ERA5 single levels Monthly means](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form))
> select *2m temperature*. Make sure you rename your file to *ERA5_REANALYSIS_air_temperature_200306.nc*
> 
> - Inspect the metadata of the new retrieved file
> - Visualize the *2m temperature* with Python (using a similar script as for the total precipitation).
>
> > ## Solution with Python
> > ~~~
> > dset = xr.open_dataset("~/Downloads/ERA5_REANALYSIS_air_temperature_200306.nc")
> > print(dset)
> >
> > import matplotlib.pyplot as plt
> > import cartopy.crs as ccrs
> >
> > fig = plt.figure(figsize=[12,5])
> >
> > # 111 means 1 row, 1 col and index 1
> > ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
> > 
> > dset['t2m'].plot(ax=ax,  cmap='jet',
> >                    transform=ccrs.PlateCarree())
> > ax.coastlines()
> > 
> > plt.show()
> > ~~~
> > {: .language-python}
> > <img src="../fig/python-t2m.png" width="80%" />
> {: .solution}
>
{: .challenge}


> ## What is 2m temperature?
> We selected [ERA5 monthly averaged data on single levels from 1979 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form) so we expected to get surface variables only. 
> In fact, we get all the variables on a single level and usually close to the surface. Here *2m temperature* is computed as 
> the temperature at a reference height (2 metres). This corresponds to the [surface air temperature](https://ane4bf-datap1.s3.eu-west-1.amazonaws.com/wmod8_gcos/s3fs-public/surface_temp_ecv_factsheet_201905.pdf?Yq5rPAs1YJ2iYVCutXWLnG_lTV.pRDb6). 
>
{: .callout}

### Change projection

It is very often convenient to visualize using a different projection than the original data:

~~~
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=[12,5])

# 111 means 1 row, 1 col and index 1
ax = fig.add_subplot(111, projection = ccrs.Orthographic(central_longitude=20, central_latitude=40))

dset['t2m'].plot(ax=ax,  cmap='jet',
                   transform=ccrs.PlateCarree())
ax.coastlines()

plt.show()
~~~
{: .language-python}

<img src="../fig/python-t2m-ortho.png" width="50%" />

### CMIP5 monthly data on single levels

Let's have a look at CMIP 5 climate data. 

#### Retrieve precipitation

We will retrieve precipitation from [CMIP5 monthly data on single levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip5-monthly-single-levels?tab=form).


As you can see that you have the choice between several models, experiments and ensemble members.


<img src="../fig/CMIP5_models.png" width="60%" />

> ## CMIP5 models
>  CMIP5 (Coupled Model Intercomparison Project Phase 5) had the following objectives:
> - evaluate how realistic the models are in simulating the recent past,
> - provide projections of future climate change on two time scales, near term (out to about 2035) and long term (out to 2100 and beyond), and
> - understand some of the factors responsible for differences in model projections, including quantifying some key feedbacks such as those involving clouds and the carbon cycle. 
>
> 20 climate modeling groups from around the world participated to CMIP5.
> All the datasets are freely available from different repositories. For more information look [here](https://esgf-node.llnl.gov/projects/esgf-llnl/).
{: .callout}

We will choose NorESM1-M (Norwegian Earth System Model 1 - medium resolution) based on the [Norwegian Earth System Model](https://no.wikipedia.org/wiki/NorESM).

Please note that it is very common to analyze several models instead of one to run statistical
analysis.

> ## CMIP5 Ensemble member
>
> Many CMIP5 experiments, the so-called ensemble calculations, were calculated using several initial 
> states, initialisation methods or physics details. Ensemble calculations facilitate quantifying the 
> variability of simulation data concerning a single model.
> 
> In the CMIP5 project, ensemble members are named in the rip-nomenclature, *r* for realization, 
> *i* for initialisation and *p* for physics, followed by an integer, e.g. r1i1p1. 
> For more information look at [Experiments, ensembles, variable names and other centralized properties](https://portal.enes.org/data/enes-model-data/cmip5/datastructure).
>
{: .callout}

Select:
- **Model**: NorESM1-M (NCC, Norway)
- **Experiment**: historical
- **Ensemble**: r1i1p1
- **Period**: 185001-200512

We rename the downloaded filename to `pr_Amon_NorESM1-M_historical_r1i1p1_185001-200512.nc`.

Let's open this NetCDF file and check its metadata:

~~~
dset = xr.open_dataset("~/Downloads/pr_Amon_NorESM1-M_historical_r1i1p1_185001-200512.nc")
print(dset)
~~~
{: .language-python}

~~~
<xarray.Dataset>
Dimensions:    (bnds: 2, lat: 96, lon: 144, time: 1872)
Coordinates:
  * time       (time) object 1850-01-16 12:00:00 ... 2005-12-16 12:00:00
  * lat        (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 86.21 88.11 90.0
  * lon        (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
Dimensions without coordinates: bnds
Data variables:
    time_bnds  (time, bnds) object ...
    lat_bnds   (lat, bnds) float64 ...
    lon_bnds   (lon, bnds) float64 ...
    pr         (time, lat, lon) float32 ...
Attributes:
    institution:            Norwegian Climate Centre
    institute_id:           NCC
    experiment_id:          historical
    source:                 NorESM1-M 2011  atmosphere: CAM-Oslo (CAM4-Oslo-n...
    model_id:               NorESM1-M
    forcing:                GHG, SA, Oz, Sl, Vl, BC, OC
    parent_experiment_id:   piControl
    parent_experiment_rip:  r1i1p1
    branch_time:            255135.0
    contact:                Please send any requests or bug reports to noresm...
    initialization_method:  1
    physics_version:        1
    tracking_id:            5ccde64e-cfe8-47f6-9de8-9ea1621e7781
    product:                output
    experiment:             historical
    frequency:              mon
    creation_date:          2011-06-01T05:45:35Z
    history:                2011-06-01T05:45:35Z CMOR rewrote data to comply ...
    Conventions:            CF-1.4
    project_id:             CMIP5
    table_id:               Table Amon (27 April 2011) a5a1c518f52ae340313ba0...
    title:                  NorESM1-M model output prepared for CMIP5 historical
    parent_experiment:      pre-industrial control
    modeling_realm:         atmos
    realization:            1
    cmor_version:           2.6.0
~~~
{: .output}

This file contains monthly averaged data from January 1850 to December 2005. 
In CMIP the variable name for precipitation flux is called **pr**, so let's look at the metadata:

~~~
print(dset.pr)
~~~
{: .language-python}

> ## Note
> The notation **dset.pr** is equivalent to **dset['pr']**.
>
{: .callout}

~~~
<xarray.DataArray 'pr' (time: 1872, lat: 96, lon: 144)>
[25878528 values with dtype=float32]
Coordinates:
  * time     (time) object 1850-01-16 12:00:00 ... 2005-12-16 12:00:00
  * lat      (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 84.32 86.21 88.11 90.0
  * lon      (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
Attributes:
    standard_name:     precipitation_flux
    long_name:         Precipitation
    comment:           at surface; includes both liquid and solid phases from...
    units:             kg m-2 s-1
    original_name:     PRECT
    cell_methods:      time: mean
    cell_measures:     area: areacella
    history:           2011-06-01T05:45:35Z altered by CMOR: Converted type f...
    associated_files:  baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation...
~~~
{: .output}

The unit is: **kg m-2 s-1**. We want to convert the units from kg m-2 s-1 to something that we are 
a little more familiar with like mm day-1 or m day-1 (metre per day) that is what we had with ERA5.

To do this, consider that 1 kg of rain water spread over 1 m2 of surface is 1 mm in thickness and 
that there are 86400 seconds in one day. Therefore, 1 kg m-2 s-1 = 86400 mm day-1 or 86.4 m day-1.

So we can go ahead and multiply that array by 86.4 and update the units attribute accordingly:

~~~
dset.pr.data = dset.pr.data * 86.4
dset.pr.attrs['units'] = 'm/day' 
~~~
{: .language-python}

Then we can select the data for June 2003 and plot the precipitation field:

~~~
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=[12,5])

# 111 means 1 row, 1 col and index 1
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

dset['pr'].sel(time='200306').plot(ax=ax, cmap='jet',
                   transform=ccrs.PlateCarree())
ax.coastlines()

plt.show()
~~~
{: .language-python}

<img src="../fig/CMIP5_pr.png" width="80%" />

To select June 2003, we used xarray select `sel`. This is a very powerful tool with which you can 
specify a particular value you wish to select. You can also add a method such as `nearest` to select
the closest point to a given value. You can even select all the values inside a range (inclusive) with `slice`:

~~~
dset['pr'].sel(time=slice('200306', '200406', 12))
~~~
{: .language-python}

This command first takes the month of June 2003, then *jumps* 12 months and takes the month of June 2004.

~~~
<xarray.DataArray 'pr' (time: 2, lat: 96, lon: 144)>
array([[[2.114823e-05, 2.114823e-05, ..., 2.114823e-05, 2.114823e-05],
        [3.787203e-05, 3.682408e-05, ..., 3.772908e-05, 3.680853e-05],
        ...,
        [7.738324e-04, 8.255294e-04, ..., 7.871171e-04, 8.004216e-04],
        [6.984189e-04, 6.986369e-04, ..., 6.984310e-04, 6.983504e-04]],

       [[1.847185e-04, 1.847185e-04, ..., 1.847185e-04, 1.847185e-04],
        [8.476275e-05, 8.299961e-05, ..., 9.207508e-05, 8.666257e-05],
        ...,
        [4.763984e-04, 4.645830e-04, ..., 5.200990e-04, 4.929897e-04],
        [5.676933e-04, 5.677207e-04, ..., 5.673288e-04, 5.674717e-04]]],
      dtype=float32)
Coordinates:
  * time     (time) object 2003-06-16 00:00:00 2004-06-16 00:00:00
  * lat      (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 84.32 86.21 88.11 90.0
  * lon      (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
Attributes:
    standard_name:     precipitation_flux
    long_name:         Precipitation
    comment:           at surface; includes both liquid and solid phases from...
    units:             m day-1
    original_name:     PRECT
    cell_methods:      time: mean
    cell_measures:     area: areacella
    history:           2011-06-01T05:45:35Z altered by CMOR: Converted type f...
    associated_files:  baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation...
~~~
{: .output}


See [here](http://xarray.pydata.org/en/stable/indexing.html)
for more information.

> ## Remark
> We selected one year (2003) and one month (June) from both ERA5 and CMIP5 but 
> only data from re-analysis (ERA5) corresponds to the actual month of June 2003.
> Data from the climate model (CMIP5 historical) is only "one realization" of a month of June,
> typical of present day conditions, but it cannot be considered as the actual weather at that date.
> To be more realistic, climate data has to be considered over a much longer period of time. For instance,
> we could easily compute (for both ERA5 and CMIP5) the average of the month of June between 1988 and 2018 (spanning 30 years) to
> have a more reliable results. However, as you (may) have noticed, the horizontal resolution of ERA5 (1/4 x 1/4 degrees) is much
> higher than that of the CMIP data (about 2 x 2 degrees) and therefore there is much more variability/details in the re-analysis data than with NorESM.
>
{: .callout}

> ## Plot surface air temperature with CMIP5 (June 2003)
> 
> When searching for [CMIP5 monthly data on single levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip5-monthly-single-levels?tab=form)
> you will see that you have the choice between several models and ensemble members.
> Select:
> - **Model**: NorESM1-M (NCC, Norway)
> - **Ensemble**: r1i1p1
>
> > ## Solution with Python
> > - Retrieve a new file with *2m temperature*
> > - rename the retrieved file to **tas_Amon_NorESM1-M_historical_r1i1p1_185001-200512.nc**
> > 
> > ~~~
> > dset = xr.open_dataset("~/Downloads/tas_Amon_NorESM1-M_historical_r1i1p1_185001-200512.nc")
> > print(dset)
> > ~~~
> > {: .language-python}
> > 
> > ~~~
> > <xarray.Dataset>
> > Dimensions:    (bnds: 2, lat: 96, lon: 144, time: 1872)
> > Coordinates:
> >   * time       (time) object 1850-01-16 12:00:00 ... 2005-12-16 12:00:00
> >   * lat        (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 86.21 88.11 90.0
> >   * lon        (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
> >     height     float64 ...
> > Dimensions without coordinates: bnds
> > Data variables:
> >     time_bnds  (time, bnds) object ...
> >     lat_bnds   (lat, bnds) float64 ...
> >     lon_bnds   (lon, bnds) float64 ...
> >     tas        (time, lat, lon) float32 ...
> > Attributes:
> >     institution:            Norwegian Climate Centre
> >     institute_id:           NCC
> >     experiment_id:          historical
> >     source:                 NorESM1-M 2011  atmosphere: CAM-Oslo (CAM4-Oslo-n...
> >     model_id:               NorESM1-M
> >     forcing:                GHG, SA, Oz, Sl, Vl, BC, OC
> >     parent_experiment_id:   piControl
> >     parent_experiment_rip:  r1i1p1
> >     branch_time:            255135.0
> >     contact:                Please send any requests or bug reports to noresm...
> >     initialization_method:  1
> >     physics_version:        1
> >     tracking_id:            c1dd6def-d613-43ab-a8b6-f4c80738f53b
> >     product:                output
> >     experiment:             historical
> >     frequency:              mon
> >     creation_date:          2011-06-01T03:52:42Z
> >     history:                2011-06-01T03:52:42Z CMOR rewrote data to comply ...
> >     Conventions:            CF-1.4
> >     project_id:             CMIP5
> >     table_id:               Table Amon (27 April 2011) a5a1c518f52ae340313ba0...
> >     title:                  NorESM1-M model output prepared for CMIP5 historical
> >     parent_experiment:      pre-industrial control
> >     modeling_realm:         atmos
> >     realization:            1
> >     cmor_version:           2.6.0
> > ~~~
> > {: .output}
> >
> > - the name of the variable is **tas** (**Near-Surface Air Temperature**).
> > We can print metadata for **tas**:
> > 
> > ~~~
> > dset.tas
> > ~~~
> > {: .language-python}
> > 
> > You can use **dset['tas']** or **dset.tas**; the syntax is different but meaning is the same. 
> > 
> > ~~~
> > <xarray.DataArray 'tas' (time: 1872, lat: 96, lon: 144)>
> > [25878528 values with dtype=float32]
> > Coordinates:
> >  * time     (time) object 1850-01-16 12:00:00 ... 2005-12-16 12:00:00
> >  * lat      (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 84.32 86.21 88.11 90.0
> >  * lon      (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
> >    height   float64 ...
> > Attributes:
> >    standard_name:     air_temperature
> >    long_name:         Near-Surface Air Temperature
> >    units:             K
> >    original_name:     TREFHT
> >    cell_methods:      time: mean
> >    cell_measures:     area: areacella
> >    history:           2011-06-01T03:52:41Z altered by CMOR: Treated scalar d...
> >    associated_files:  baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation...
> > ~~~
> > {: .output}
> > 
> > ~~~
> > import matplotlib.pyplot as plt
> > import cartopy.crs as ccrs
> > 
> > fig = plt.figure(figsize=[12,5])
> > 
> > # 111 means 1 row, 1 col and index 1
> > ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
> > 
> > dset['tas'].sel(time='200306').plot(ax=ax, cmap='jet',
> >                    transform=ccrs.PlateCarree())
> > ax.coastlines()
> > 
> > plt.show()
> > ~~~
> > {: .language-python}
> >
> > <img src="../fig/as_Amon_NorESM1-M_historical_r1i1p1_200306.png" width="80%" />
> > 
> {: .solution}
{: .challenge}

## How to open several files with `xarray`?

In our last exercise, we downloaded two files, containing temperature and precipitation.

With `xarray`, it is possible to open these two files at the same time and get one unique view e.g.
very much like if we had one file only:

~~~
import xarray as xr

# use open_mfdataset to open alll netCDF files 
# contained in the directory called cmip5
dset = xr.open_mfdataset("cmip5/*.nc")
print(dset)
~~~
{: .python}

~~~
    <xarray.Dataset>
    Dimensions:    (bnds: 2, lat: 96, lon: 144, time: 1872)
    Coordinates:
      * time       (time) object 1850-01-16 12:00:00 ... 2005-12-16 12:00:00
      * lat        (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 86.21 88.11 90.0
      * lon        (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
        height     float64 ...
    Dimensions without coordinates: bnds
    Data variables:
        time_bnds  (time, bnds) object dask.array<shape=(1872, 2), chunksize=(1872, 2)>
        lat_bnds   (lat, bnds) float64 dask.array<shape=(96, 2), chunksize=(96, 2)>
        lon_bnds   (lon, bnds) float64 dask.array<shape=(144, 2), chunksize=(144, 2)>
        pr         (time, lat, lon) float32 dask.array<shape=(1872, 96, 144), chunksize=(1872, 96, 144)>
        tas        (time, lat, lon) float32 dask.array<shape=(1872, 96, 144), chunksize=(1872, 96, 144)>
    Attributes:
        institution:            Norwegian Climate Centre
        institute_id:           NCC
        experiment_id:          historical
        source:                 NorESM1-M 2011  atmosphere: CAM-Oslo (CAM4-Oslo-n...
        model_id:               NorESM1-M
        forcing:                GHG, SA, Oz, Sl, Vl, BC, OC
        parent_experiment_id:   piControl
        parent_experiment_rip:  r1i1p1
        branch_time:            255135.0
        contact:                Please send any requests or bug reports to noresm...
        initialization_method:  1
        physics_version:        1
        tracking_id:            5ccde64e-cfe8-47f6-9de8-9ea1621e7781
        product:                output
        experiment:             historical
        frequency:              mon
        creation_date:          2011-06-01T05:45:35Z
        history:                2011-06-01T05:45:35Z CMOR rewrote data to comply ...
        Conventions:            CF-1.4
        project_id:             CMIP5
        table_id:               Table Amon (27 April 2011) a5a1c518f52ae340313ba0...
        title:                  NorESM1-M model output prepared for CMIP5 historical
        parent_experiment:      pre-industrial control
        modeling_realm:         atmos
        realization:            1
        cmor_version:           2.6.0
~~~
{: .output}


## How to use `xarray` to get metadata?

### Get the list of variables


~~~
for varname, variable in dset.items():
    print(varname)
~~~
{: .python}

~~~
    time_bnds
    lat_bnds
    lon_bnds
    pr
    tas
~~~
{: .output}

### Create a python dictionary with variable and long name


~~~
dict_variables = {}
for varname, variable in dset.items():
    if len(dset[varname].attrs) > 0:
        dict_variables[dset[varname].attrs['long_name']] = varname
print(dict_variables)
print(list(dict_variables.keys()))
~~~
{: .python}

~~~
    {'Precipitation': 'pr', 'Near-Surface Air Temperature': 'tas'}
    ['Precipitation', 'Near-Surface Air Temperature']
~~~
{: .output}

### Select temperature (tas)

- two notations for accessing variables


~~~
dset.tas
~~~
{: .python}



~~~
    <xarray.DataArray 'tas' (time: 1872, lat: 96, lon: 144)>
    dask.array<shape=(1872, 96, 144), dtype=float32, chunksize=(1872, 96, 144)>
    Coordinates:
      * time     (time) object 1850-01-16 12:00:00 ... 2005-12-16 12:00:00
      * lat      (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 84.32 86.21 88.11 90.0
      * lon      (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
        height   float64 ...
    Attributes:
        standard_name:     air_temperature
        long_name:         Near-Surface Air Temperature
        units:             K
        original_name:     TREFHT
        cell_methods:      time: mean
        cell_measures:     area: areacella
        history:           2011-06-01T03:52:41Z altered by CMOR: Treated scalar d...
        associated_files:  baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation...
~~~
{: .output}



~~~
dset['tas']
~~~
{: .language-python}



~~~
    <xarray.DataArray 'tas' (time: 1872, lat: 96, lon: 144)>
    dask.array<shape=(1872, 96, 144), dtype=float32, chunksize=(1872, 96, 144)>
    Coordinates:
      * time     (time) object 1850-01-16 12:00:00 ... 2005-12-16 12:00:00
      * lat      (lat) float64 -90.0 -88.11 -86.21 -84.32 ... 84.32 86.21 88.11 90.0
      * lon      (lon) float64 0.0 2.5 5.0 7.5 10.0 ... 350.0 352.5 355.0 357.5
        height   float64 ...
    Attributes:
        standard_name:     air_temperature
        long_name:         Near-Surface Air Temperature
        units:             K
        original_name:     TREFHT
        cell_methods:      time: mean
        cell_measures:     area: areacella
        history:           2011-06-01T03:52:41Z altered by CMOR: Treated scalar d...
        associated_files:  baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation...
~~~
{: .output}


### Get attributes


~~~
dset['tas'].attrs['long_name']
~~~
{: .python}


~~~
    'Near-Surface Air Temperature'
~~~
{: .output}



~~~
dset['tas'].attrs['units']
~~~
{: .python}



~~~
    'K'
~~~
{: .output}


### How to create a function to generate a plot for a given date and given variable?

Instead of copy-paste the same code several times to plot different variables and dates,
it is common to define a function with the date and variable as parameters:

~~~
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def generate_plot(date, variable):
    
    fig = plt.figure(figsize=[12,5])

    # 111 means 1 row, 1 col and index 1
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    
    dset[variable].sel(time=date).plot(cmap='jet', 
                                      transform=ccrs.PlateCarree())
    ax.coastlines(color='white', linewidth=2.)

~~~
{: .python}

### Call `generate_plot` function to plot 2m temperature


~~~
generate_plot(dset.time.values.tolist()[0], 'tas')
~~~
{: .python}


![png](../fig/2m_temperature_plot.png)


## Plotting on-demand with Jupyter widgets

Our goal in this section is to learn to create a user interface where one can select the variable and date to
plot.


### Create a widget to select available variables


~~~
%matplotlib inline
import ipywidgets as widgets
select_variable = widgets.Select(
    options=['2m temperature', 'precipitation'],
    value='2m temperature',
    rows=2,
    description='Variable:',
    disabled=False
)

display(select_variable)
~~~
{. .python}


### Create a widget to select date (year and month)


~~~
%matplotlib inline
import ipywidgets as widgets
select_date = widgets.Dropdown(
    options=dset.time.values.tolist(),
    rows=2,
    description='Date:',
    disabled=False
)

display(select_date)
~~~
{: .python}



### Use widgets to plot the chosen variable at the chosen date


~~~
import ipywidgets as widgets
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# widget for variables
list_variables = list(dict_variables.keys())
select_variable = widgets.Select(
    options=list_variables,
    value=list_variables[0],
    rows=2,
    description='Variable:',
    disabled=False
)

# Widget for date
select_date = widgets.Dropdown(
    options=dset.time.values.tolist(),
    rows=2,
    description='Date:',
    disabled=False
)

# generate plot 
def generate_plot(date, variable):
    
    fig = plt.figure(figsize=[12,5])

    # 111 means 1 row, 1 col and index 1
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    
    dset[dict_variables[variable]].sel(time=date).plot(cmap='jet', 
                                      transform=ccrs.PlateCarree())
    ax.coastlines(color='white', linewidth=2.)


interact_plot = widgets.interact(generate_plot, date = select_date, variable = select_variable);
~~~
{: .python}

## Satellite Data Preprocessing


Satellites are spaceborne platforms that carry earth observation (EO) sensors and use remote sensing technologies to collect information about the earth. There are numerous satellites orbiting the earth and collecting vast amount of data that can be utilized for:

- Agriculture: crop monitoring
- Forestry: forestry planning and prevention of illegal logging
- Fishing: prevention of illegal fishing
- Energy: pipeline and right-of-way monitoring
- Insurance: infrastructure integrity monitoring
- Land use: infrastructure planning and monitoring of building activity
- Sea traffic: iceberg monitoring, oil spills detection
- Security: coastal traffic monitoring
- Disaster response: fast response to natural catastrophes

Depending on the type of sensor equipment onboard a satellite, the format of data collected by a satellite can be one of:

- GeoTIFF
- HDF
- NDF
- NITF
- NetCDF
- XML

These variety of data formats have specialized tools used for reading and processing them.

In this section, we will go through how to use  `Xarray` which can be used for reading, processing and writing most of the common satellite data formats including NetCDF, HDF, XML and GeoTIFF.

~~~
# Install required libraries
#!pip install -q earthpy rioxarray cftime h5pyd Bottleneck

# Import packages
import warnings
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import earthpy as et
import random
import numpy as np
import pandas as pd
warnings.simplefilter('ignore')
# pip install earthpy
import earthpy as et
et.data.path = "."
et.data.get_data()
# Available Datasets
et.data.get_data('cs-test-landsat')
et.data.get_data('cold-springs-modis-h4')
et.data.get_data('cold-springs-fire')

~~~
{: .python}

~~~
$ wget -q -O world_boundaries.zip 'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/world-administrative-boundaries/exports/shp'
$ unzip -q world_boundaries.zip
~~~
{: .bash}


### Reading HDF Data

Reading a HDF file using Xarray (Rioxarray) requires a complete installation of GDAL which is not obtainable using pip install rasterio.
~~~
ds = rxr.open_rasterio(hdf_file, engine='h5netcdf')
~~~

See [here](https://rasterio.readthedocs.io/en/latest/installation.html#advanced-installation) for a complete installation of GDAL which is required for reading HDF files

### GeoTIFF

~~~
tif_file = 'cs-test-landsat/LC08_L1TP_034032_20160621_20170221_01_T1_sr_band4.tif'

ds = xr.open_dataset(tif_file)
~~~
{: .python}

### Reading Shapefile

A Coordinate reference system (CRS) defines, with the help of coordinates, how the two-dimensional, projected map is related to real locations on the earth.

There are two different types of coordinate reference systems:

Geographic Coordinate Systems
Projected Coordinate Systems
The most popular geographic coordinate system is called WGS84 (EPSG:4326). It comprises of lines of latitude that run parallel to the equator and divide the earth into 180 equally spaced sections from North to South (or South to North) and lines of longitude that run perpendicular to the equator and converge at the poles.

Using the geographic coordinate system, we have a grid of lines dividing the earth into squares of 1 degrees (appprox. 111 Km) resolution that cover approximately 12363.365 square kilometres at the equator — a good start, but not very useful for determining the location of anything within that square.

We can divide a map grid into sub-units of degrees such as 0.1 degrees (11.1 km), 0.01 (1.11 km) etc. See [here](https://www.usna.edu/Users/oceano/pguth/md_help/html/approx_equivalents.htm) for approximate metric equivalent of degrees

A shapefile is a simple, nontopological format for storing the geometric location and attribute information of geographic features. Geographic features in a shapefile can be represented by points, lines, or polygons (areas).

~~~
shapefile_path = "data/world-administrative-boundaries.shp"
gdf = gpd.read_file(shapefile_path)
gdf[gdf['name']=='Ethiopia'].geometry.plot()

~~~
{: .python}

#### Cropping

We can use a shapefile to crop a dataarray. Lets download a global temperature data from [Physical Sciences Laboratory (PSL)](https://psl.noaa.gov/)

~~~
%%bash
wget -q 'https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.2013.nc'

~~~
{: .python}

~~~
ds = xr.open_dataset('data/tmin.2013.nc')
ds
~~~
{: .python}

~~~
<xarray.Dataset>
Dimensions:  (lat: 360, lon: 720, time: 365)
Coordinates:
  * lat      (lat) float32 89.75 89.25 88.75 88.25 ... -88.75 -89.25 -89.75
  * lon      (lon) float32 0.25 0.75 1.25 1.75 2.25 ... 358.2 358.8 359.2 359.8
  * time     (time) datetime64[ns] 2013-01-01 2013-01-02 ... 2013-12-31
Data variables:
    tmin     (time, lat, lon) float32 ...
Attributes:
    Conventions:    CF-1.0
    Source:         ftp://ftp.cpc.ncep.noaa.gov/precip/wd52ws/global_temp/
    version:        V1.0
    title:          CPC GLOBAL TEMP V1.0
    dataset_title:  CPC GLOBAL TEMP
    history:        Updated 2019-09-10 20:13:53
    References:     https://www.psl.noaa.gov/data/gridded/data.cpc.globaltemp...
~~~
{: .output}

Here we visualize a map of global minimum temperature for the day 2013-01-01

~~~
ds.sel(time='2013-01-01')['tmin'].plot();
~~~

![](../fig/map_global_tmin_2013.png)


To visualize the map for a single country, all we need to do is to use the shapifle for the country( e.g Ethiopia) to crop the global map

~~~
country = 'Ethiopia'
da  = ds.sel(time='2013-01-01')['tmin']

# we specify the CRS as EPSG:4326
ds.rio.write_crs('epsg:4326', inplace=True)

selected_country_shapefile = gdf[gdf['name']==country].geometry
cropped_ds = ds.rio.clip(selected_country_shapefile)
cropped_ds.sel(time='2013-10-10')['tmin'].plot()
~~~
{: .python}

![](../fig/map_ethiopia_tmin_2013.png)

Do you notice that the map is highly pixelated? This is because the resolution of the data is low

~~~
cropped_ds['lon'][1] - cropped_ds['lon'][0]
~~~
{: .python}

~~~
    
<xarray.DataArray 'lon' ()>
array(0.5, dtype=float32)
Coordinates:
    spatial_ref  int64 0
~~~
{. output}


~~~
cropped_ds['lat'][0] - cropped_ds['lat'][1]
~~~
{: .python

~~~
<xarray.DataArray 'lat' ()>
array(0.5, dtype=float32)
Coordinates:
    spatial_ref  int64 0
~~~
{: .output}

The resolution on both the longitude and latitudes axes is 0.5 degrees which means each pixel covers an approximate area of 3,080 square kilometres. Hence, we can increase/decrease the spatial resolution by resampling.

#### Resampling (Upsampling/Downsampling)

To upsample an xarray DataArray, you can use the resample() method to change the frequency of the time dimension or the interp() method to increase the resolution of spatial dimensions. Here are examples of both approaches:

Many atimes geospatial datasets would need to be resampled wither by increasing the resolution (upsampling) or decreasing the resolution (downsampling) on either the spatial dimensions (lat/lon) or time dimension.

##### Time Dimension

For example, a dataset may contain weekly records and we desire to resample to daily or monthly. These type of resampling involves the time dimension and we use the resample() method of Xarray

~~~
cropped_ds
~~~
{: .python}

~~~
<xarray.Dataset>
Dimensions:      (lat: 19, lon: 24, time: 365)
Coordinates:
  * lat          (lat) float32 13.75 13.25 12.75 12.25 ... 6.25 5.75 5.25 4.75
  * lon          (lon) float32 2.75 3.25 3.75 4.25 ... 12.75 13.25 13.75 14.25
  * time         (time) datetime64[ns] 2013-01-01 2013-01-02 ... 2013-12-31
    spatial_ref  int64 0
Data variables:
    tmin         (time, lat, lon) float32 nan nan nan nan ... nan nan nan nan
Attributes:
    Conventions:    CF-1.0
    Source:         ftp://ftp.cpc.ncep.noaa.gov/precip/wd52ws/global_temp/
    version:        V1.0
    title:          CPC GLOBAL TEMP V1.0
    dataset_title:  CPC GLOBAL TEMP
    history:        Updated 2019-09-10 20:13:53
    References:     https://www.psl.noaa.gov/data/gridded/data.cpc.globaltemp...
~~~
{: .output}

Note how the size of the time dimension changes from 365 (days) to 53 (weeks)

~~~
weekly_ds = cropped_ds.resample(time='W').interpolate('linear')
weekly_ds
~~~
{: .python}

~~~
xarray.Dataset>
Dimensions:      (lat: 22, lon: 29, time: 53)
Coordinates:
  * lat          (lat) float32 14.25 13.75 13.25 12.75 ... 5.25 4.75 4.25 3.75
  * lon          (lon) float32 33.25 33.75 34.25 34.75 ... 46.25 46.75 47.25
    spatial_ref  int64 0
  * time         (time) datetime64[ns] 2013-01-06 2013-01-13 ... 2014-01-05
Data variables:
    tmin         (time, lat, lon) float64 nan nan nan nan ... nan nan nan nan
Attributes:
    Conventions:    CF-1.0
    Source:         ftp://ftp.cpc.ncep.noaa.gov/precip/wd52ws/global_temp/
    version:        V1.0
    title:          CPC GLOBAL TEMP V1.0
    dataset_title:  CPC GLOBAL TEMP
    history:        Updated 2019-09-10 20:13:53
    References:     https://www.psl.noaa.gov/data/gridded/data.cpc.globaltemp...
~~~
{: .output}

##### Spatial Dimensions

As we saw earlier that our dataset has spatial resolutions of 0.5 degrees (equivalent to 55km) along both latitude and longitude diemensions. This may be too low for some applications.

We can increase/decrease the spatial resolutions of Xarray datarrays by resampling using the `interp()` method


~~~
cropped_ds.sel(time='2013-10-10')['tmin'].plot();
~~~
{: .python}

![](../fig/map_ethiopia_tmin_xresampled.png)


~~~
country_ds = ds.sel(lat=slice(16, 2), lon=slice(33, 48))
new_lon = np.linspace(country_ds.lon[0], country_ds.lon[-1], country_ds.dims["lon"] * 4)
new_lat = np.linspace(country_ds.lat[0], country_ds.lat[-1], country_ds.dims["lat"] * 4)

higher_resolution_ds = country_ds.interp(lat=new_lat, lon=new_lon)

# we specify the CRS as EPSG:4326
higher_resolution_ds.rio.write_crs('epsg:4326', inplace=True)

selected_country_shapefile = gdf[gdf['name']==country].geometry
higher_resolution_cropped_ds = higher_resolution_ds.rio.clip(selected_country_shapefile)
higher_resolution_cropped_ds.sel(time='2013-10-10')['tmin'].plot();
~~~
{: .python}

![](../fig/map_ethiopia_tmin_2013_high_resolution.png)

##### Nan-Filling

Geo-spatial datasets often contain gaps or missing values, which can impact the analysis and visualization of the data. To address these missing values, several methods can be employed:

1. **fillna**: This method involves filling the missing values with a specified constant or a calculated value. It allows for directly replacing the missing data with a chosen value.

2. **ffill (forward fill)**: With this method, missing values are filled by propagating the last known value forward along the dataset. It is useful when the data has a trend or pattern that can be carried forward.

3. **bfill (backward fill)**: In contrast to ffill, bfill fills missing values by propagating the next known value backward along the dataset. This method is beneficial when the data exhibits a pattern that can be extended backward.

Lets also syntheticaly add more variables to our dataset, so that when we sample we can retrieve four variables including `tmin`, `tmax`, `pressure` and `wind_speed`

~~~
time_dim, lat_dime, lon_dim = higher_resolution_ds['tmin'].shape

higher_resolution_ds['tmax'] = higher_resolution_ds['tmin'] * (1+np.random.rand(time_dim, lat_dime, lon_dim))
higher_resolution_ds['pressure'] = higher_resolution_ds['tmin'] * (np.cos(np.random.rand(time_dim, lat_dime, lon_dim)))
higher_resolution_ds['wind_speed'] = higher_resolution_ds['tmin'] * (np.sin(np.random.rand(time_dim, lat_dime, lon_dim)))
higher_resolution_ds
~~~
{: .python}

~~~
<xarray.Dataset>
Dimensions:      (time: 365, lat: 112, lon: 120)
Coordinates:
  * time         (time) datetime64[ns] 2013-01-01 2013-01-02 ... 2013-12-31
  * lat          (lat) float64 15.75 15.63 15.51 15.39 ... 2.493 2.372 2.25
  * lon          (lon) float64 33.25 33.37 33.49 33.62 ... 47.51 47.63 47.75
    spatial_ref  int64 0
Data variables:
    tmin         (time, lat, lon) float64 18.42 18.56 18.7 18.83 ... nan nan nan
    tmax         (time, lat, lon) float64 30.9 24.76 23.39 37.38 ... nan nan nan
    pressure     (time, lat, lon) float64 16.33 14.39 12.63 ... nan nan nan
    wind_speed   (time, lat, lon) float64 5.568 12.15 1.572 ... nan nan nan
Attributes:
    Conventions:    CF-1.0
    Source:         ftp://ftp.cpc.ncep.noaa.gov/precip/wd52ws/global_temp/
    version:        V1.0
    title:          CPC GLOBAL TEMP V1.0
    dataset_title:  CPC GLOBAL TEMP
    history:        Updated 2019-09-10 20:13:53
    References:     https://www.psl.noaa.gov/data/gridded/data.cpc.globaltemp...
~~~
{: .output}

~~~
lat = np.random.uniform(2, 16, 100).tolist()
lon = np.random.uniform(33, 48, 100).tolist()

dates = pd.date_range(start='2013-01-01', end='2013-12-31', freq='D')
time = random.sample(list(dates), 100)
data = pd.DataFrame({'lat': lat, 'lon': lon, 'time':time})
data
~~~
{: .python}

~~~
          lat        lon       time
0    3.903489  37.250512 2013-03-23
1    6.592635  46.735060 2013-04-22
2   10.021625  39.345432 2013-05-03
3   13.983304  45.731566 2013-02-06
4    6.105293  35.664100 2013-05-26
..        ...        ...        ...
95   7.741817  44.221370 2013-09-08
96   2.491084  38.560937 2013-04-24
97  14.669314  44.535139 2013-01-06
98   4.273452  38.612193 2013-02-19
99   2.363141  38.282740 2013-12-22

[100 rows x 3 columns]
~~~
{: .output}

~~~
dataset = higher_resolution_ds.sel(
    lat=xr.DataArray(data["lat"], dims="z"),
    lon=xr.DataArray(data["lon"], dims="z"),
    time=xr.DataArray(data["time"], dims="z"),
    method='nearest'
)
dataset
~~~
{: .python}

~~~
<xarray.Dataset>
Dimensions:      (z: 100)
Coordinates:
    time         (z) datetime64[ns] 2013-10-12 2013-06-12 ... 2013-07-24
    lat          (z) float64 11.25 4.074 14.9 5.534 ... 5.899 2.615 12.83 11.49
    lon          (z) float64 46.04 33.74 41.66 43.61 ... 41.29 36.54 47.75 38.61
    spatial_ref  int64 0
  * z            (z) int64 0 1 2 3 4 5 6 7 8 9 ... 90 91 92 93 94 95 96 97 98 99
Data variables:
    tmin         (z) float64 nan 20.13 nan 22.28 22.68 ... 20.63 23.46 nan 14.38
    tmax         (z) float64 nan 34.22 nan 23.21 40.23 ... 36.35 27.96 nan 20.45
    pressure     (z) float64 nan 16.73 nan 19.47 22.27 ... 12.89 15.94 nan 13.99
    wind_speed   (z) float64 nan 5.641 nan 16.39 8.23 ... 16.23 9.55 nan 9.868
Attributes:
    Conventions:    CF-1.0
    Source:         ftp://ftp.cpc.ncep.noaa.gov/precip/wd52ws/global_temp/
    version:        V1.0
    title:          CPC GLOBAL TEMP V1.0
    dataset_title:  CPC GLOBAL TEMP
    history:        Updated 2019-09-10 20:13:53
    References:     https://www.psl.noaa.gov/data/gridded/data.cpc.globaltemp...
~~~
{: .output}

~~~
dataset_df = dataset.to_dataframe().reset_index()[['tmin', 'tmax', 'pressure', 'wind_speed']].dropna(how="all")
dataset_df
~~~
{: .python}


~~~
      tmin       tmax     pressure   wind_speed
1   20.134108  34.216812  16.731225    5.640895
3   22.279944  23.211976  19.469766   16.390574
4   22.675937  40.226501  22.273046    8.230098
5   18.908172  26.481716  16.766963   14.388920
6   23.847037  42.906728  17.507207   13.681055
..        ...        ...        ...         ...
94  23.607236  46.954574  23.605141    5.492570
95  19.637835  35.957441  13.043338   16.507524
96  20.629210  36.350215  12.889924   16.226357
97  23.455832  27.960924  15.941195    9.550094
99  14.376755  20.446728  13.990531    9.867571

~~~
{: .output}


~~~
dataset_df.to_csv('geospatial_dataset.csv',index=0)
~~~
{: .python}

The above steps show how to read and preprocess satellite data using Xarray and other anciliary libraries.


