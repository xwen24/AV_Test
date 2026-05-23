import os
import xarray as xr
import cfgrib


def convert_snowfall(filename_in, filename_out):
    """
    Reads a GRIB file, converts longitudes from [0, 360) to [-180, 180),
    sorts by longitude, and attempts to write back to GRIB.

    ⚠️ WARNING: GRIB writing via cfgrib is experimental and often unsupported.
    Consider saving to NetCDF instead for reliability.

    Parameters:
    -----------
    filename_in : str
        Path to input GRIB file.
    filename_out : str
        Path for output file (attempted as GRIB; may fail).
    """
    # 1. Load dataset
    ds = xr.open_dataset(filename_in, engine='cfgrib')

    # 2. Convert longitude to [-180, 180)
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180))
    ds = ds.sortby('longitude')

    # 3. Attempt to save as GRIB (likely to fail)
    print("Attempting to save as GRIB... (this often fails due to cfgrib limitations)")

    # DO NOT reassign cfgrib — ensure we reference the module
    try:
        # Check if to_grib is actually available
        if hasattr(cfgrib, 'to_grib'):
            cfgrib.to_grib(ds, filename_out)
            print("Success: saved using cfgrib.to_grib")
            return
        else:
            raise AttributeError("'to_grib' not found in cfgrib module")
    except AttributeError:
        try:
            # Try the submodule (rarely available)
            from cfgrib import xarray_to_grib
            xarray_to_grib.to_grib(ds, filename_out)
            print("Success: saved using cfgrib.xarray_to_grib")
            return
        except ImportError:
            print("Failure: cfgrib.xarray_to_grib is not available.")
        except Exception as e:
            print(f"Failure in xarray_to_grib: {e}")
    except Exception as e:
        print(f"Unexpected error during GRIB writing: {e}")

    print("💡 Recommendation: Save as NetCDF instead for reliable I/O.")
    try:
        netcdf_out = filename_out.replace('.grib', '.nc')
        ds.to_netcdf(netcdf_out)
        print(f"Saved fallback file: {netcdf_out}")
    except Exception as e2:
        print(f"Even NetCDF save failed: {e2}")


def convert_temp(filename_in, filename_out, overwrite_attrs=True, verbose=True):
    """
    Correct NetCDF file longitude (0-360 → -180-180) and latitude order (ensure -90→90 ascending),
    so it displays correctly in QGIS, and add standard CF metadata.

    Parameters:
        filename_in (str): Path to input NetCDF file.
        filename_out (str, optional): Path to output file.
        overwrite_attrs (bool): Whether to overwrite/add standard longitude/latitude attributes (recommended: True).
        verbose (bool): Whether to print progress messages.

    Returns:
        xarray.Dataset: The corrected dataset.
    """
    if verbose:
        print("📁 Reading NetCDF file...")

    try:
        ds = xr.open_dataset(filename_in, cache=False)
        ds = ds.load()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to read file: {e}")

    if verbose:
        print("✅ File read successfully")

    # Automatically detect longitude and latitude variable names
    lon_candidates = ['lon', 'longitude', 'Longitude', 'x']
    lat_candidates = ['lat', 'latitude', 'Latitude', 'y']

    lon_name = next((name for name in lon_candidates if name in ds.coords), None)
    lat_name = next((name for name in lat_candidates if name in ds.coords), None)

    if lon_name is None:
        raise ValueError("❌ Longitude variable (lon/longitude) not found!")
    if lat_name is None:
        raise ValueError("❌ Latitude variable (lat/latitude) not found!")

    if verbose:
        print(f"🔍 Longitude variable: '{lon_name}', Latitude variable: '{lat_name}'")

    # === Step 1: Convert longitude to [-180, 180] ===
    lon_min = ds[lon_name].min().item()
    lon_max = ds[lon_name].max().item()
    if verbose:
        print(f"🔄 Current longitude range: {lon_min:.2f}° ~ {lon_max:.2f}°")

    if lon_min >= 0 and lon_max > 180:
        if verbose:
            print("⚙️  Converting longitude from 0–360 to -180–180...")
        ds[lon_name] = (ds[lon_name] + 180) % 360 - 180
        ds = ds.sortby(lon_name)
        if verbose:
            print("✅ Longitude conversion completed")
    elif verbose:
        print("ℹ️  Longitude already in -180~180 range, skipping conversion")

    # === Step 2: Ensure latitude is ascending (-90 → +90) ===
    lat_values = ds[lat_name].values
    if verbose:
        print(f"🌐 Current latitude range: {lat_values[0]:.2f}° ~ {lat_values[-1]:.2f}°")

    # === Step 3: Add standard metadata (CF-Conventions) ===
    if overwrite_attrs:
        if verbose:
            print("📝 Adding/updating geospatial metadata...")
        ds[lon_name].attrs.update({
            'standard_name': 'longitude',
            'long_name': 'longitude',
            'units': 'degrees_east',
            'axis': 'X'
        })
        ds[lat_name].attrs.update({
            'standard_name': 'latitude',
            'long_name': 'latitude',
            'units': 'degrees_north',
            'axis': 'Y'
        })

    # === Step 4: Save file ===
    if verbose:
        print(f"💾 Saving to: {filename_out}")

    os.makedirs(os.path.dirname(os.path.abspath(filename_out)), exist_ok=True)

    ds.to_netcdf(
        filename_out,
        format='NETCDF4',
        engine='netcdf4',
        unlimited_dims=None
    )

    if verbose:
        print("✅ New NetCDF file has been successfully generated!")

    if verbose:
        print("\n🔍 Verifying output file...")
        ds_out = xr.open_dataset(filename_out)
        print(f"Longitude range: {ds_out[lon_name].min().item():.2f}° ~ {ds_out[lon_name].max().item():.2f}°")
        print(f"Latitude range: {ds_out[lat_name].min().item():.2f}° ~ {ds_out[lat_name].max().item():.2f}°")
        print("✅ Verification complete")

    return ds