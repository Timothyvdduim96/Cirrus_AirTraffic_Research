# Cirrus_AirTraffic_Research

The following Python scripts and respective classes are contained:

- AIRCRAFT.py
  - flight_analysis: imports and processes air traffic data up to 2018, computes ATD, outputs Fig 1.2, 2.2, 3.1 from report
  - flights_after_2018: subsets flights (presumably) flying over the ROI for the years 2019 and 2020
- ML_model.py
  - ML_model: creates model array for Jan 2015 ('model_arrays' method), does a train-test split and evaluates LR and RF model.
	      Also includes CALIPSO vs ERA5 temp validation. Outputs Fig 3.5, 3.6, 4.14 from report
- METEOSAT.py
  - meteosat: processes L2 Meteosat files and outputs L3 gridded product for cirrus cloud cover, cot and other params.
  - METEOSAT_analysis: analyzes cirrus cover over March 2015 from Meteosat product together with air traffic. Validates Meteosat
                       product with CALIPSO. Outputs Fig 3.8, 4.8, 4.9, 4.10, 4.11, 4.13 and Table 4.1
- miscellaneous.py
  - ERA5_reanalysis: used to extract ECMWF ERA5 reanalysis data through API requests
  - miscellaneous: set of auxiliary functions used in other files like unit converters, color generator, time rounders, dataframe 
                   formatting, data binners etc.
- time_series.py
  - time_series: creates long-term time series (2015-2020), splits data into subsaturated and supersaturated air and into daytime
                 and nightime, deduces ATD for 2019 and 2020 using the 'flights_after_2018' method and the 'flight_analysis' method.
                 Outputs Fig 3.4, 3.7, 4.3, 4.4, 4.5, 4.6, 4.7, 4.12
- vert_profiles.py
  - vertical_profiles: generates time series on varying pressure levels for meteo and cirrus data. Outputs Fig 4.1 and 4.2b
  - flights_vert_res: generates time series on varying pressure levels air traffic data. Outputs Fig 4.2a
- CALIPSO.py
  - hdf4_files: facilitates the data extraction from HDF4 files
  - CALIPSO_analysis: imports and processes CALIPSO data and generates CALIPSO gridded L3 products like cirrus cover, cloud temperature
    and optical depth (latter only for 5km product)
  - CALIPSO_visuals: generates positional heat map and time map for CALIPSO overpasses for each month (input). Used in time_series to
                     generate Fig 3.7 and 4.12
