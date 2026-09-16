---
title: Data Sources
teaching: 1
exercises: 0
questions:
- "Understanding data sources"
- "How to get data from online sources"
- "How to retrieve dataset with the Toolbox?"
objectives:
- "Brief overview of various data souces"
- "Discuss the benefits and disadvantages of each."
- "Learn to combine Climate data with your own research topic"
- "Learn how to manipulate netCDF data within the CDS Toolbox"
keypoints:
- "Essential libaries for data online data sources"
- "Data retrieval from the CDS Toolbox"
- "netCDF and GRIB data formats"
---

# Climate and Weather Data Sources


## 1) CHIRPS 

**CHIRPS (Climate Hazards Group Infrared Precipitation with Stations)** offers a comprehensive global precipitation dataset, combining satellite observations with ground-based station data to deliver accurate rainfall estimates. 


CHIRPS stands out for its high spatial and temporal resolution of rainfall estimates, accommodating regional variations in dataset relationships. Despite its strengths, the dataset has limitations, including its focus solely on rainfall, limited coverage to land areas between 50S-50N, and its primary application in drought and environmental monitoring.



### Main Features

With spatial resolution at approximately 5 km, CHIRPS is instrumental in various applications. It aids in disaster management by monitoring droughts and floods, supports climate research for studying climate change impacts, facilitates agricultural planning, and assists in water resource management.

  - It's a high-resolution gridded precipitation dataset.
  - Combines satellite observations with rain gauge data.
  - Provides daily, monthly, and seasonal precipitation estimates.
  - Long-term Data: Available from 1981 onwards, useful for trend analysis.
  - Frequent Updates: Data updated in near real-time for timely information.
  
### Data Source

Spearheaded by the Climate Hazards Group and USGS scientists, CHIRPS receives support from USAID, NASA, and NOAA.

  - **Organization**: Climate Hazards Group and USGS scientists—supported by funding from USAID, NASA, and NOAA
  - **Website**: [CHG Official Site](https://www.chc.ucsb.edu/data/chirps)
  - **Data Access**: [CHIRPS Data Access](https://data.chc.ucsb.edu/products/CHIRPS-2.0/)

### Description

The dataset features high-resolution gridded precipitation data, providing daily, monthly, and seasonal estimates. Covering temporal data from 1981 onwards, CHIRPS facilitates trend analysis and is regularly updated in near real-time, ensuring timely information availability.

  - **Temporal Coverage**: 1981 to present
  - **Spatial Coverage**: 50°S to 50°N, all longitudes
  - **Resolution**: 
    - **Temporal**: 6-hourly, Daily, Pentad, Monthly, Seasonal
    - **Spatial**: 0.05° (approx. 5 km) and 0.1°
 
    
### Applications
  - **Disaster Management**: Crucial for monitoring droughts and floods.
  - **Climate Research**: Supports climate change research and impact assessments.
  - **Agricultural Monitoring**: Informs agricultural planning and risk management.
  - **Water resources**: Valuable for water resource management.

### Key Strengths
  - High spatial and temporal resolution of rainfall estimates
  - Accounts for regional variations in the relationships between different datasets
  - Relatively long-term record (1981-present) with nearly immediate updates of the dataset

### Key Limitations
  - Rainfall only; does not include snow or ice
  - Land-only due to its heavy reliance on station data and not available outside of 50S-50N
  - Intended for drought and environmental monitoring; less known is its performance as a dataset to study low-frequency climate variability and trends

### Downloading CHIRPS Data  
  - [Jupyter Note Book Script](https://github.com/mesfind/ml-physical/blob/gh-pages/code/data_source/CHIRPS_global_daily_Precip.ipynb)


## 2) CPC Global Unified Gauge-Based Analysis of Daily Precipitation

  - The CPC Global Unified Gauge-Based Analysis of Daily Precipitation is a comprehensive dataset developed by the **Climate Prediction Center (CPC)** of the **National Oceanic and Atmospheric Administration (NOAA)** by taking advantage of the **optimal interpolation (OI)** (> 30,000 gauges (optimal interp. with orographic effects).

  - This dataset provides **global daily** precipitation estimates based on **gauge observations**, offering valuable insights for various applications in **climate research**, **hydrology**, and **weather forecasting**.

   - **Temporal Coverage/Duration**: 1979/01/01 to Present. 

   - **Time Step**: Daily, Monthly

   - **Spatial Resolution**: 0.5x0.5 deg

   - **Missing data**:  flagged with a value of -9.96921e+36f.

   - **Source**: https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html

**Key Strengths**: 

  - High station density

**Key Limitations**: 

  - Quality of the gauge-based analysis is poor over tropical Africa and Antarctica (gauge network density).

### Downloading CPC Data  
 - [Jupyter Note Book Script](https://github.com/mesfind/ml-physical/blob/gh-pages/code/data_source/CPC_global_daily_Precip.ipynb)

---

## 3) International Research Institute for Climate and Society (IRI), Data Library (DL) 

- a collection of datasets, collected from various sources, designed to make them more accessible for the library’s users. (Bluementhal, 2004).

- It includes wide range of climate datasets:
     
  **Seasonal forecasts**
  
  **Historical observations**
  
  **Climate model outputs**
  
  **Reanalysis data**
  
  **Global and regional coverage**
  
  **Freely available for download**

- Source: https://iridl.ldeo.columbia.edu/

Example1: 

<img src="../fig/IRI_CPC.png" width="80%"/>

Example2: 

<img src="../fig/IRI_ARC.png" width="80%"/>

Hands-on:

- Download  daily TAMSAT rainfall data for Ethiopia at least for one year from the Reading University dataset.

---

## 4) Copernicus Climate Data Store (CDS)

- This is a web portal providing a single point of access to a wide range of information.

- This is a service operated by the [European Centre for Medium-range Weather Forecasts (ECMWF)](https://www.ecmwf.int/) on behalf of the European Union. 

- Free and Open Access Climate Data: C3S offers free and open access to a vast collection of climate data and information.

- This includes:
  -  observations (i.e., in-situ measurements, remote sensing data, etc.),
  -  historical climate data records,
  -  estimates of Essential Climate Variables (ECVs) derived from Earth observations,
  -  global and regional climate reanalyses of past observations,
  -  seasonal forecasts and
  -  climate projections.

![](../fig/C3S_frontpage.png)

### Climate Data Store (CDS) Registration

To be able to use CDS services, you need to [register](https://cds.climate.copernicus.eu/user/login?destination=%2F%23!%2Fhome). Registration to the Climate Data Store (CDS) is free as well as access to climate data.
Before starting, and once registred, login to the Climate Data Store (CDS).

![](../fig/CDS_login.png)


### Retrieve Climate data with CDS API

Using CDS web interface is very useful when you need to retrieve small amount of data and you do not need to customize your request. However, it is often very useful to retrieve climate data directly on the computer where you need to run your postprocessing workflow.

In that case, you can use the CDS API (Application Programming Interface) to retrieve Climate data directly in Python from the Climate Data Store.

We will be using `cdsapi` python package.

### Get your API key

- Make sure you login to the [Climate Data Store](https://cds.climate.copernicus.eu/#!/home)

- Click on your username (top right of the main page) to get your API key.
 
![](../fig/get_your_cds_api_key.png)

- Copy the code displayed beside, in the file $HOME/.cdsapirc

~~~
url: https://cds.climate.copernicus.eu/api/v2
key: UID:KEY
~~~
{: .bash}

Where UID is your `uid` and KEY your API key. See [documentation](https://cds.climate.copernicus.eu/api-how-to) to get your API and related information.

### Install the CDS API client
~~~
pip3 install cdsapi
~~~

### Use CDS API

Once the CDS API client is installed, it can be used to request data from the datasets listed in the CDS catalogue. It is necessary to agree to the Terms of Use of every datasets that you intend to download.

Attached to each dataset download form, the button Show API Request displays the python code to be used. The request can be formatted using the interactive form. The api call must follow the syntax:

~~~
import cdsapi
c = cdsapi.Client()

c.retrieve("dataset-short-name", 
           {... sub-selection request ...}, 
           "target-file")
~~~
{: .python}

For instance to retrieve the same ERA5 dataset e.g. near surface air temperature for June 2003:

![](../fig/CDSAPI_t2m_ERA5.png)

Let’s try it:

~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type':'monthly_averaged_reanalysis',
        'variable':'2m_temperature',
        'year':'2003',
        'month':'06',
        'time':'00:00',
        'format':'netcdf'
    },
    'download.nc')
~~~
{: .python}

### Geographical subset

~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {      
        'area'          : [60, -10, 50, 2], # North, West, South, East. Default: global
        'product_type':'monthly_averaged_reanalysis',
        'variable':'2m_temperature',
        'year':'2003',
        'month':'06',
        'time':'00:00',
        'format':'netcdf'
    },
    'download_small_area.nc')
~~~
{: .python}

### Change horizontal resolution

For instance to get a coarser resolution:
~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {      
        'area'          : [60, -10, 50, 2], # North, West, South, East. Default: global
        'grid'          : [1.0, 1.0], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'product_type':'monthly_averaged_reanalysis',
        'variable':'2m_temperature',
        'year':'2003',
        'month':'06',
        'time':'00:00',
        'format':'netcdf'
    },
    'download_small.nc')
~~~
{: .python}

More information can be found [here](https://confluence.ecmwf.int/display/CKB/C3S+ERA5%3A+Web+API+to+CDS+API).

### To download CMIP 5 Climate data via CDS API

~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'projections-cmip5-monthly-single-levels',
    {
        'variable':'2m_temperature',
        'model':'noresm1_m',
        'experiment':'historical',
        'ensemble_member':'r1i1p1',
        'period':'185001-200512'
    },
    'download_CMIP5.nc')
~~~
{: .python}


> ## Exercise: Download CMIP5 from Climate Data Store with `cdsapi`
> Get near surface air temperature (2m temperature) and precipitation (mean precipitation flux) in one single request and save the result in a file `cmip5_sfc_monthly_1850-200512.zip`.
> What do you get when you unzip this file?
> > ## Solution
> > 
> >  - Download the file 
> >  - Uncompress it
> >  - If you select one variable, one experiment, one model, etc., then you get one file only, and it is a netCDF file (even if it says otherwise!). As soon as you select more than one variable, or more than one experiment, etc., then you get a zip or tgz (depending on the format you chose).
> >
> > ~~~
> > import cdsapi
> > import os
> > import zipfile
> > c = cdsapi.Client()
> > c.retrieve(
> >     'projections-cmip5-monthly-single-levels', 
> >     { 
> >        'variable': ['2m_temperature',
> >       'mean_precipitation_flux'],
> >        'model': 'noresm1_m',
> >         'experiment': 'historical',
> >         'ensemble_member': 'r1i1p1',
> >         'period': '185001-200512',
> >         'format': 'tgz'
> >     },
> >     'cmip5_sfc_monthly_1850-200512.zip'
> > )
> > os.mkdir("./cmip5")
> > with zipfile.ZipFile('cmip5_sfc_monthly_1850-200512.zip', 'r') as zip_ref:
> >     zip_ref.extractall('./cmip5')
> > ~~~
> > {: .python}
> {: .solution}
{: .challenge}

---
## 5) Meteostat 


- A Python library provides a simple API for accessing **open weather** and **climate data**.

- Meteorological data provided by Meteostat (https://dev.meteostat.net) under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License. The code is licensed under the **MIT license**.

- The **historical observations** and **statistics** are collected by Meteostat from different public interfaces, most of which are governmental.

    - Deutscher Wetterdienst
    - NOAA - National Weather Service
    - NOAA - Global Historical Climatology Network
    - NOAA - Integrated Surface Database
    - Government of Canada - Open Data 
    - MET Norway 
    - European Data Portal
    - Offene Daten Österreich 


- Guide:  https://dev.meteostat.net/python/

### Installation

The Meteostat Python package is available through PyPI

~~~
$ pip install meteostat
~~~

[Hands-on Meteostat](https://github.com/mesfind/ml-physical/blob/gh-pages/code/data_source/Meteostat.ipynb)

 
![](../fig/daily_temp_addis_ababa_2013.png)

---

## 6) CliMetLab

- **CliMetLab** is Python package to support **AI/ML* activities in climate and meteorology.


- **CliMetLab** allow users to focus on science instead of technical issues such as data access and data formats.

- It is mostly intended to be used in **Jupyter notebooks**, and be interoperable with all popular data analytic packages, such as **NumPy**, **Pandas**, **Xarray**, **SciPy**, **Matplotlib**, etc.

- Datasets are automatically downloaded, cached and transform into standard Python data structures. 

- As well as machine learning frameworks, such as **TensorFlow**, **Keras** or **PyTorch**.

- CliMetLab also provides very **high-level map plotting facilities**.

- Source: https://climetlab.readthedocs.io/en/latest/index.html

![](../fig/CliMetLab.png)


TO install CliMetLab, just run the following command:

~~~
pip install climetlab
~~~

[Hands-on climetlab](https://github.com/mesfind/ml-physical/blob/gh-pages/code/data_source/climetlab.ipynb)

---

# Climate Data Processsing with Xarray

[Hands-on xarray](https://github.com/mesfind/ml-physical/blob/gh-pages/code/data_source/hans-on_xarray.ipynb)

---
 
