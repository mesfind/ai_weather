## What data format for Climate data?

Climate data can become large very quickly (as we usually need to analyze data over large period of time and covering large geographical areas) so we do not store them as text files (i.e., your usual ASCII format, csv files, tabular, etc.) to compress them as much as possible without loosing any important information.

All Climate data are stored in **binary** format and hence are not *human readable*.

Depending on the type of Climate data, you may have the choice between several data formats:

- [GRIB](https://en.wikipedia.org/wiki/GRIB)
- [NetCDF](https://en.wikipedia.org/wiki/NetCDF)

> ## Data format: GRIB versus NetCDF
>
> 
> ### NetCDF
>
> [NetCDF](https://en.wikipedia.org/wiki/NetCDF)  ([Network Common Data Form](https://www.unidata.ucar.edu/software/netcdf/)) is a set of software libraries and self-describing, machine-independent data formats that support the creation, access, and sharing of array-oriented scientific data. NetCDF is commonly used to store and distribute scientific data. 
> The NetCDF software was developed at the [Unidata Program Center](http://www.unidata.ucar.edu/publications/factsheets/current/factsheet_netcdf.pdf) in Boulder, Colorado (USA). 
> NetCDF files usually have the extension *.nc*. 
> As for the GRIB format, NetCDF files are binary and you need to use specific tools to read them. NetCDF files can also be manipulated with most programming languages (R, Python, C, Fortran, etc.).
> 
> For climate and forecast data stored in NetCDF format there are (non-mandatory) conventions on metadata ([CF Convention](http://cfconventions.org/)). 
> 
> ### GRIB
>
> [GRIB](https://en.wikipedia.org/wiki/GRIB) (GRIdded Binary or General Regularly-distributed Information in Binary form) is a file format designed for storing and distributing weather data. GRIB files are mostly used in meteorological applications. The last ECMWF re-analysis (ERA5) is natively encoded in GRIB and also in a version converted from GRIB to NetCDF. Note that due to limitations of the NetCDF specifications, the NetCDF version contains fewer parameters (variables) and incomplete metadata (information about the data). 
> As this format is not widely used there are not as many tools or programming languages supported as netCDF.
>
{: .callout}


Whenever we can, we will choose to download data in NetCDF format but we will also add links to documentation with examples using native GRIB format.


NetCDF format is a binary format and to be able to read or visualize it, we would need to use dedicated software or libraries that can handle this "special" format.

##  Radiant Earth MLHub

To automatically download the dataset, you need an API key from the [Radiant Earth MLHub](https://mlhub.earth/). This is completely free, and will give you access to a growing catalog of ML-ready remote sensing datasets.

~~~
# Set this to your API key (available for free at https://beta.source.coop/auth/registration)
RADIANT_EARTH_API_KEY = ""

data_dir = os.path.join(tempfile.gettempdir(), "cyclone_data")

datamodule = CycloneDataModule(
    root_dir=data_dir,
    seed=1337,
    batch_size=64,
    num_workers=6,
    api_key=RADIANT_EARTH_API_KEY
)

~~~
{: .python}

## [geospatial-datasets](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#geospatial-datasets)


GeoDataset is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection. Datasets containing this kind of information can be combined using IntersectionDataset and UnionDataset.


In most cases, we'll use `geopandas` to read vector files and analyze data:

~~~
import geopandas as gpd

# Load the countries dataframe using geopandas
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
countries.head()
~~~
{: .python}

We can use shapely attributes and operations to get geometries of interest

~~~
# Plot the union of all african countries
countries[countries["continent"] == "Africa"].unary_union
~~~
{: .python}



ERA5 consists of the climate variable data from 1959 to present. The ERA5 dataset is freely-accessible to the community on the Climate Data Store (CDS) hosted by the European Union's Earth observation programme, Copernicus. The users can visit https://cds.climate.copernicus.eu/api-how-to to install the CDS API key required to access their data.
The authors from [2] (WeatherBench dataset) downloaded the ERA5 data from CDS API for a subset of climate variable and preprocessed it at multiple resolutions. The data is freely accessible to anyone at https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2F.
