import calendar
import os
import gc
import glob
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from shapely.geometry import box
from shapely.errors import ShapelyError
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots
from scipy.stats import mannwhitneyu
from utilities.hyperparameters import ROAD_TYPES, THRESHOLD, SKIP_INACCESSIBLE_IDS, POI_TAGS, LABELS
from utilities.calc_gdp_per_capita import calc_gdp_per_capita
from utilities.extract_citi_names import extract_citi_names
from utilities.convert_longitude import convert_snowfall, convert_temp
from Area_Calc_4 import clean_df, load_gpkg
from Road_Calc_5 import process_gpkg_polygons, calculate_inter_number
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial
import osmnx as ox
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"
ox.settings.use_cache = True
ox.settings.requests_timeout = 300
ox.settings.log_console = False


# ==========================================
# 配置与常量
# ==========================================
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "Data/QGIS"
RESULTS_DIR = BASE_DIR / "src/Results"

FILTER_COL = "GC_POP_TOT_2025"
ID_COL = "ID_UC_G0"
NAME_COL = "GC_UCN_MAI_2025"
LAYER_NAMES = [
    "GHS_UCDB_THEME_GHSL_GLOBE_R2024A",
    "GHS_UCDB_THEME_SOCIOECONOMIC_GLOBE_R2024A"
]

PATHS = {
    "GPKG": DATA_DIR / "GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg",
    "GEOJSON": RESULTS_DIR / "Total_Dist/Built_Up",
    "FILTER": DATA_DIR / f"GHS_UCDB_GLOBE_R2024A_V1_1_{THRESHOLD}",
    "RESULTS": RESULTS_DIR / f"Global_Comp/{THRESHOLD}",
    "AREA_CSV": RESULTS_DIR / "Area_Calc/Area_Calc.csv",
    "ACCESSIBLE_ROAD_DIR": RESULTS_DIR / "Road_Calc/AV_Accessible",
    "PRECIP": DATA_DIR / "Precipitation",
    "SNOW": DATA_DIR / "Snowfall",
    "TEMP": DATA_DIR / "Temperature",
    "SLOPE_TIF": DATA_DIR / "Slope/slope_1KMmn_SRTM.tif",
    "ENTROPY": BASE_DIR / "Data/OSM",
    "POI_DIR": BASE_DIR / "Data/POI"
}

for p in [PATHS["FILTER"], PATHS["RESULTS"], PATHS["SNOW"], PATHS["TEMP"]]:
    p.mkdir(parents=True, exist_ok=True)


# ==========================================
# 通用工具函数
# ==========================================
def load_geometry_data(is_accessible: bool):
    if is_accessible:
        files = list(PATHS["GEOJSON"].glob("*.geojson"))
        if not files:
            raise FileNotFoundError("No GeoJSON files found.")

        gdf_list = []
        for f in files:
            try:
                temp = gpd.read_file(f)
                merged_geom = temp.unary_union
                city_gdf = gpd.GeoDataFrame([{'city': f.stem, 'geometry': merged_geom}], crs=temp.crs)
                gdf_list.append(city_gdf)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                continue

        gdf = pd.concat(gdf_list, ignore_index=True)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf, "city"
    else:
        gpkg_path = PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg"
        gdf = load_gpkg(gpkg_path)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf, ID_COL


def save_result(df, filename):
    name_cols = ['city', NAME_COL]
    for col in name_cols:
        if col in df.columns:
            df[col] = df[col].replace('Xiongan', "Xiong'an")

    path = PATHS["RESULTS"] / filename
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved: {path}")


def calculate_mean_raster_value(geometry, raster_path):
    """
    计算给定 geometry 在 raster_path 栅格上的平均值。
    返回 float 或 None（无有效数据）。
    """
    try:
        with rasterio.open(raster_path) as src:
            if geometry.is_empty or not geometry.is_valid:
                return None

            out_image, _ = mask(src, [geometry], crop=True, all_touched=True)
            data = out_image[0]

            nodata = src.nodata
            if nodata is not None:
                valid = data[data != nodata]
            else:
                valid = data.flatten()

            valid = valid[~np.isnan(valid)]  # 使用 numpy 的 isnan
            if valid.size == 0:
                return None
            return float(valid.mean())
    except Exception as e:
        print(f"Error computing raster stat for {raster_path}: {e}")
        return None


def compute_raster_zonal_stat(raster_path: Path, is_accessible: bool, indicator_name: str):
    """
    通用函数：对 accessible/inaccessible 城市计算栅格平均值（如 slope, tri）。
    """
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    gdf, id_col = load_geometry_data(is_accessible)

    # 获取栅格 CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # 重投影
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    results = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        entry = {id_col: row[id_col]}
        if not is_accessible:
            entry[NAME_COL] = row[NAME_COL]

        avg_val = calculate_mean_raster_value(geom, raster_path)
        entry[indicator_name] = avg_val
        results.append(entry)

    df = pd.DataFrame(results)
    if not is_accessible:
        df = clean_df(df)

    prefix = "accessible" if is_accessible else "inaccessible"
    save_result(df, f"av_{prefix}_{indicator_name}.csv")


# ==========================================
# 数据预处理
# ==========================================
def filter_inaccessible_cities():
    """
    从全球城市数据库中过滤出不可达（AV-inaccessible）的城市：
    - 人口 > THRESHOLD
    - 空间上不与任何 AV-accessible GeoJSON 区域相交
    - 排除用户指定的黑名单城市
    """
    geojson_files = list(PATHS["GEOJSON"].glob("*.geojson"))
    if not geojson_files:
        raise ValueError("No GeoJSON files found!")

    print("Merging GeoJSONs for spatial filtering...")
    gdfs = [gpd.read_file(f) for f in geojson_files]
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    union_geom = merged_gdf.unary_union

    result_counts = {}
    for layer in LAYER_NAMES:
        out_path = PATHS["FILTER"] / f"{layer}_FILTER.gpkg"
        if out_path.exists():
            print(f"Skipping existing filtered layer: {layer}")
            continue

        print(f"Processing layer: {layer}")
        gdf = load_gpkg(PATHS["GPKG"], layer=layer)

        # 确保 CRS 为 EPSG:4326
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # 清理无效几何
        gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid].copy()

        # 应用人口和空间过滤
        filtered = gdf[
            (gdf[FILTER_COL] > THRESHOLD) &
            (~gdf.geometry.intersects(union_geom))
            ].copy()

        if SKIP_INACCESSIBLE_IDS:
            original_count = len(filtered)
            filtered = filtered[
                ~filtered[ID_COL].astype(str).isin(SKIP_INACCESSIBLE_IDS)
            ]
            skipped_count = original_count - len(filtered)
            if skipped_count > 0:
                print(f"  → Skipped {skipped_count} blacklisted cities in {layer}")

        count = len(filtered)
        result_counts[layer] = count

        # 保存过滤后的 GPKG
        filtered.to_file(out_path, driver="GPKG", encoding="utf-8")
        print(f"  → Saved {count} cities to {out_path.name}")

    return result_counts


# ==========================================
# 人均GDP 计算
# ==========================================
def compute_gdp(is_accessible):
    if not is_accessible:
        gpkg_path = PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg"
        gdf = load_gpkg(gpkg_path)

        try:
            gdp_series = calc_gdp_per_capita(str(gpkg_path))
            gdf["gdp"] = gdp_series

            res = gdf[[ID_COL, NAME_COL, "gdp"]].replace([np.inf, -np.inf], np.nan)
            res = res.dropna(subset=["gdp"])
            res = res[res["gdp"] > 0]

            save_result(res, "av_inaccessible_gdp.csv")

        except Exception as e:
            print(f"计算不可达城市 GDP 时发生错误: {e}")
            raise e

    else:
        cities = extract_citi_names("src/Results/Total_Dist/Corp/")
        data = []

        for city in cities:
            geo_name = "Sanya" if city == "Sanya, China" else city
            gj_path = PATHS["GEOJSON"] / f"{geo_name}.geojson"

            try:
                val = calc_gdp_per_capita(POLY_GEOJSON=str(gj_path)).iloc[0]

                if pd.isna(val) or val <= 0:
                    raise ValueError

                if pd.notna(val):
                    if city == "Sanya, China": city = "Sanya"
                    data.append({"city": city, "gdp": val})
            except Exception as e:
                print(f"Skipping {city}: {e}")
                continue

        save_result(pd.DataFrame(data), "av_accessible_gdp.csv")


# ==========================================
# GDP总量 计算
# ==========================================
def compute_gdp_sum(is_accessible):
    if not is_accessible:
        gpkg_path = PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg"
        gdf = load_gpkg(gpkg_path)

        try:
            # 1. 计算人均 GDP
            gdp_per_capita_series = calc_gdp_per_capita(str(gpkg_path))
            gdf["gdp_per_capita"] = gdp_per_capita_series

            # 2. 计算 GDP 总量
            gdf["gdp_sum"] = gdf["gdp_per_capita"] * gdf["GH_POP_TOT_2020"]

            # 3. 清理和筛选有效数据
            res = gdf[[ID_COL, NAME_COL, "gdp_sum"]].replace([np.inf, -np.inf], np.nan)
            res = res.dropna(subset=["gdp_sum"])
            res = res[res["gdp_sum"] > 0]

            save_result(res, "av_inaccessible_gdp_sum.csv")
        except Exception as e:
            print(f"计算不可达城市 GDP 总量时发生错误: {e}")
            raise e

    else:
        cities = extract_citi_names("src/Results/Total_Dist/Corp/")
        data = []
        area_csv = pd.read_csv(PATHS["AREA_CSV"])

        CITY_NAME_MAP = {"Sanya, China": "Sanya"}

        for raw_city in cities:
            city = CITY_NAME_MAP.get(raw_city, raw_city)
            gj_path = PATHS["GEOJSON"] / f"{city}.geojson"

            try:
                # 1. 计算人均 GDP
                gdp_series = calc_gdp_per_capita(POLY_GEOJSON=str(gj_path))
                if gdp_series.empty:
                    raise ValueError("GDP per capita result is empty")
                gdp_per_capita_val = gdp_series.iloc[0]
                if pd.isna(gdp_per_capita_val) or gdp_per_capita_val <= 0:
                    raise ValueError(f"Invalid GDP per capita: {gdp_per_capita_val}")

                # 2. 获取总人口
                city_row = area_csv[area_csv['city'] == city]
                if city_row.empty:
                    raise ValueError(f"City '{city}' not found in area CSV")
                population_val = city_row['estimated_population'].iloc[0]

                # 3. 计算 GDP 总量
                gdp_sum_val = gdp_per_capita_val * population_val
                if pd.notna(gdp_sum_val) and gdp_sum_val > 0:
                    data.append({"city": city, "gdp_sum": gdp_sum_val})
                else:
                    raise ValueError("Computed GDP sum is invalid or non-positive")

            except Exception as e:
                print(f"Skipping {raw_city} (mapped to '{city}'): {e}")
                continue

        save_result(pd.DataFrame(data), "av_accessible_gdp_sum.csv")


# ==========================================
# 人口密度计算
# ==========================================
def compute_pop_density(is_accessible):
    if not is_accessible:
        gdf = load_gpkg(PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg")
        gdf["pop_density"] = (gdf["GH_POP_TOT_2020"] / gdf["GC_UCA_KM2_2025"])
        res = gdf[[ID_COL, NAME_COL, "pop_density"]].replace([np.inf, -np.inf], np.nan).dropna()
        save_result(res, "av_inaccessible_pop_density.csv")
    else:
        df = pd.read_csv(PATHS["AREA_CSV"])
        df = df[~df["city"].isin(["Ezhou", "Grand Rapids Mn", "Jiaxing"])].copy()
        df["pop_density"] = (df["estimated_population"] / df["intersect_area_km2"])
        save_result(df[["city", "pop_density"]],
                    "av_accessible_pop_density.csv")


# ==========================================
# 人口总量计算
# ==========================================
def compute_pop_size(is_accessible):
    if not is_accessible:
        # Load global urban centres (unfiltered by AV deployment)
        gdf = load_gpkg(PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg")
        # Use total population directly (no division by area)
        gdf["pop_size"] = gdf["GH_POP_TOT_2020"]
        # Keep only valid entries (non-null, finite)
        res = gdf[[ID_COL, NAME_COL, "pop_size"]].replace([np.inf, -np.inf], np.nan).dropna()
        save_result(res, "av_inaccessible_pop_size.csv")
    else:
        # Load AV-deployed cities with estimated population and intersection area
        df = pd.read_csv(PATHS["AREA_CSV"])
        # Remove problematic cities
        df = df[~df["city"].isin(["Ezhou", "Grand Rapids Mn", "Jiaxing"])].copy()
        # Use estimated population as total population
        df["pop_size"] = df["estimated_population"]
        save_result(df[["city", "pop_size"]],
                    "av_accessible_pop_size.csv")


# ==========================================
# 年降雨量量计算
# ==========================================
def compute_annual_prep(is_accessible):
    """
    Computes total annual precipitation (sum of all daily bands).
    """
    gdf, id_col = load_geometry_data(is_accessible)
    if gdf.crs is None:
        print("⚠️ Warning: GDF 丢失坐标系，默认设为 EPSG:4326")
        gdf.set_crs("EPSG:4326", inplace=True)

    # 搜索文件
    nc_files = sorted(glob.glob(str(PATHS["PRECIP"] / "precip_*.nc")))

    if not nc_files:
        print(f"❌ 错误: 未找到任何 precip*.nc 文件。")
        return

    results = gdf[[id_col, NAME_COL]].copy() if not is_accessible else gdf[[id_col]].copy()

    for tif in nc_files:
        filename = Path(tif).name

        match = re.search(r"(\d{4})", filename)
        if match:
            year = match.group(1)
        else:
            print(f"⚠️ Warning: 无法从文件名 {filename} 中提取年份")
            year = "unknown"

        print(f"Processing Total Annual Precipitation for Year: {year}")

        with rasterio.open(tif) as src:
            # 1. 检查波段数
            if src.count < 300:
                print(
                    f"Warning: File {tif} has only {src.count} bands. Requires daily data (approx 365 bands). Skipping.")
                results[f"annual_prep_{year}"] = None
                continue

            # 2. 读取所有波段数据
            # data shape is (bands, height, width)
            data = src.read()

            # 获取 NoData 值
            nodata = src.nodata if src.nodata is not None else -9999

            # 3. 核心计算逻辑：计算年降雨总量
            # 创建掩膜：标记有效数据（不等于 NoData 的部分）
            valid_mask = (data != nodata)

            # 将 NoData 区域临时设为 0，以免影响求和
            # 注意：这里使用 copy 避免修改原始 data 导致后续 mask 判断错误，或者直接操作 masked array
            data_filled = np.where(valid_mask, data, 0)

            # 沿时间轴 (axis=0) 求和，得到每个像素点的“年总降雨量”
            annual_precip_layer = np.sum(data_filled, axis=0).astype(float)

            # 恢复 NoData 区域
            # 如果某像素在第0天是 NoData，通常意味着该像素在整个时间序列都是无效的（如海洋或边界外）
            mask_2d = (data[0] == nodata)
            annual_precip_layer[mask_2d] = -9999  # 设置为新的 nodata 值

            # 4. 处理坐标系
            target_crs = src.crs

            # 如果读取到的 CRS 是 None (NetCDF常见情况)，手动指定为 EPSG:4326
            if target_crs is None:
                print(f"⚠️ 文件 {Path(tif).name} 无 CRS，默认使用 EPSG:4326")
                target_crs = "EPSG:4326"

            try:
                # 使用 target_crs 而不是 src.crs
                if gdf.crs != target_crs:
                    geom_to_use = gdf.to_crs(target_crs).geometry
                else:
                    geom_to_use = gdf.geometry
            except Exception as e:
                print(f"❌ 投影转换出错 ({year}): {e}")
                results[f"annual_prep_{year}"] = None
                continue

            # 5. 使用计算好的图层进行 Zonal Stats
            try:
                stats = zonal_stats(
                    geom_to_use,
                    annual_precip_layer,
                    affine=src.transform,
                    stats=["mean"],
                    all_touched=True,
                    nodata=-9999
                )
                vals = [s["mean"] for s in stats]
            except Exception as e:
                print(f"❌ Zonal Stats 计算出错 ({year}): {e}")
                vals = None

        # 保存该年份的结果
        results[f"annual_prep_{year}"] = vals

    # 后处理逻辑
    if is_accessible:
        results = results.groupby(id_col, as_index=False).first()
    else:
        results = clean_df(results)

    prefix = "accessible" if is_accessible else "inaccessible"

    output_filename = f"av_{prefix}_annual_prep.csv"
    save_result(results, output_filename)


# ==========================================
# 极端降水计算
# ==========================================
def compute_extreme_prep(is_accessible, threshold_mm=20):
    gdf, id_col = load_geometry_data(is_accessible)
    if gdf.crs is None:
        print("⚠️ Warning: GDF 丢失坐标系，默认设为 EPSG:4326")
        gdf.set_crs("EPSG:4326", inplace=True)

    # 搜索文件
    nc_files = sorted(glob.glob(str(PATHS["PRECIP"] / "precip_*.nc")))

    if not nc_files:
        print(f"❌ 错误: 未找到任何 precip*.nc 文件。")
        return

    results = gdf[[id_col, NAME_COL]].copy() if not is_accessible else gdf[[id_col]].copy()

    for tif in nc_files:
        filename = Path(tif).name

        match = re.search(r"(\d{4})", filename)
        if match:
            year = match.group(1)
        else:
            print(f"⚠️ Warning: 无法从文件名 {filename} 中提取年份")
            year = "unknown"

        print(f"Processing Precipitation (Threshold {threshold_mm}mm) for Year: {year}")

        with rasterio.open(tif) as src:
            # 1. 检查波段数
            if src.count < 300:
                print(
                    f"Warning: File {tif} has only {src.count} bands. Requires daily data (approx 365 bands). Skipping.")
                results[f"extreme_prep_{year}"] = None
                continue

            # 2. 读取所有波段数据
            data = src.read()

            # 获取 NoData 值
            nodata = src.nodata if src.nodata is not None else -9999

            # 3. 核心计算逻辑：生成统计图层
            valid_mask = (data != nodata)
            heavy_rain_hits = (data >= threshold_mm) & valid_mask

            # 沿时间轴 (axis=0) 求和，得到每个像素点的“达标天数”
            r20_layer = np.sum(heavy_rain_hits, axis=0).astype(float)

            # 恢复 NoData 区域
            mask_2d = (data[0] == nodata)
            r20_layer[mask_2d] = -9999  # 设置为新的 nodata 值

            # 4. 处理坐标系
            target_crs = src.crs

            # 如果读取到的 CRS 是 None (NetCDF常见情况)，手动指定为 EPSG:4326
            if target_crs is None:
                print(f"⚠️ 文件 {Path(tif).name} 无 CRS，默认使用 EPSG:4326")
                target_crs = "EPSG:4326"

            try:
                # 使用 target_crs 而不是 src.crs
                if gdf.crs != target_crs:
                    geom_to_use = gdf.to_crs(target_crs).geometry
                else:
                    geom_to_use = gdf.geometry
            except Exception as e:
                print(f"❌ 投影转换出错 ({year}): {e}")
                results[f"extreme_prep_{year}"] = None
                continue

            # 5. 使用计算好的图层进行 Zonal Stats
            try:
                stats = zonal_stats(
                    geom_to_use,
                    r20_layer,
                    affine=src.transform,
                    stats=["mean"],
                    all_touched=True,
                    nodata=-9999
                )
                vals = [s["mean"] for s in stats]
            except Exception as e:
                print(f"❌ Zonal Stats 计算出错 ({year}): {e}")
                vals = None

        # 保存该年份的结果
        results[f"extreme_prep_{year}"] = vals

    # 后处理逻辑
    if is_accessible:
        results = results.groupby(id_col, as_index=False).first()
    else:
        results = clean_df(results)

    prefix = "accessible" if is_accessible else "inaccessible"

    output_filename = f"av_{prefix}_extreme_prep.csv"
    save_result(results, output_filename)


# ==========================================
# 降雪计算
# ==========================================
def compute_snowfall(is_accessible):
    gdf, id_col = load_geometry_data(is_accessible)

    print("Loading Snowfall GRIB...")
    ds = xr.open_dataset(PATHS["SNOW"] / "snowfall.grib", engine='cfgrib')
    var_name = next(v for v in ['sf', 'snowfall', 'tp', 'sd'] if v in ds)

    da = ds[var_name].sortby('time')
    if da.sizes['time'] == 60:
        da = da.assign_coords(time=pd.date_range('2016-01-01', periods=60, freq='MS'))
    else:
        da = da.sel(time=slice("2016-01-01", "2020-12-31"))

    da = da.rio.write_crs("EPSG:4326")
    monthly = [(da.isel(time=i) * calendar.monthrange(t.year, t.month)[1]).expand_dims(time=[t])
               for i, t in enumerate(pd.to_datetime(da.time.values))]
    da_annual = xr.concat(monthly, dim='time').groupby('time.year').sum(dim='time').rio.write_crs("EPSG:4326")

    target_years = range(2016, 2021)
    out_data = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        entry = {id_col: row[id_col]}
        if not is_accessible: entry[NAME_COL] = row[NAME_COL]

        try:
            if geom is None or geom.is_empty: raise ValueError
            masked = da_annual.rio.clip([geom], crs="EPSG:4326", all_touched=True)
            for y in target_years:
                val = float(masked.sel(year=y).mean().item()) * 1000 if y in masked.year else np.nan
                entry[f"snowfall_{y}"] = val
        except:
            for y in target_years: entry[f"snowfall_{y}"] = np.nan

        out_data.append(entry)

    df = pd.DataFrame(out_data)
    if not is_accessible: df = clean_df(df)
    prefix = "accessible" if is_accessible else "inaccessible"
    save_result(df, f"av_{prefix}_snowfall.csv")


# ==========================================
# 气温计算
# ==========================================
def compute_temperature(is_accessible, stat_type, threshold=None):
    gdf, id_col = load_geometry_data(is_accessible)
    prefix = f"{stat_type}_temp_"
    nc_files = sorted(glob.glob(str(PATHS["TEMP"] / f"{prefix}*.nc")))

    result_df = gdf[[id_col, NAME_COL]].copy() if not is_accessible else gdf[[id_col]].copy()

    # Determine the actual threshold to use
    if threshold is not None:
        current_threshold = threshold
    else:
        # Set default defaults if not provided
        if stat_type == 'max':
            current_threshold = 35  # Default hot day threshold
        elif stat_type == 'min':
            current_threshold = 0  # Default frost day threshold
        else:
            current_threshold = None

    for n_file in nc_files:
        fname = Path(n_file).name
        year = re.search(r'(\d{4})', fname).group(1) if re.search(r'(\d{4})', fname) else "unknown"

        # Create a dynamic column name based on the operation
        if stat_type == 'max' and current_threshold is not None:
            col_name = f"{prefix}gt{int(current_threshold)}_{year}"  # e.g., max_temp_gt35_2020
        elif stat_type == 'min' and current_threshold is not None:
            col_name = f"{prefix}lt{int(current_threshold)}_{year}"  # e.g., min_temp_lt0_2020
        else:
            col_name = f"{prefix}{year}"

        print(f"Processing {stat_type} for Year: {year} | Threshold: {current_threshold}")

        try:
            with xr.open_dataset(n_file) as ds:
                da = ds[list(ds.data_vars)[0]]
                spatial_stat_op = stat_type

                if 'time' in da.dims:
                    if stat_type == 'max':
                        # --- Calculate Extreme Heat Days (> threshold) ---
                        # Using the passed threshold variable
                        hot_days_mask = xr.where((da.notnull()) & (da > current_threshold), 1, 0)
                        da = hot_days_mask.sum(dim='time')
                        da = da.astype('float32')
                        spatial_stat_op = 'mean'  # We want the average number of days per polygon

                    elif stat_type == 'min':
                        # --- Calculate Frost Days (< threshold) ---
                        # Using the passed threshold variable
                        frost_days_mask = xr.where((da.notnull()) & (da < current_threshold), 1, 0)
                        da = frost_days_mask.sum(dim='time')
                        da = da.astype('float32')
                        spatial_stat_op = 'mean'  # We want the average number of days per polygon

                    else:
                        # Standard mean calculation
                        if da.sizes['time'] > 1:
                            da = da.mean(dim='time')
                        else:
                            da = da.squeeze('time')
                        spatial_stat_op = 'mean'

                if da.rio.crs is None: da.rio.write_crs("EPSG:4326", inplace=True)

                data_array = da.values.astype(np.float32)
                affine = da.rio.transform()

                stats = zonal_stats(
                    gdf, data_array, affine=affine,
                    stats=spatial_stat_op,
                    nodata=np.nan, all_touched=True
                )

                result_df[col_name] = [s[spatial_stat_op] for s in stats]

        except Exception as e:
            print(f"Error {fname}: {e}")

    if is_accessible:
        result_df = result_df.groupby(id_col, as_index=False).first()
    else:
        result_df = clean_df(result_df)

    # Filter columns to save
    final_cols = [id_col] + ([NAME_COL] if not is_accessible else []) + \
                 sorted([c for c in result_df.columns if prefix in c])

    file_prefix = "accessible" if is_accessible else "inaccessible"

    # Update filename to reflect threshold if applicable
    thresh_str = ""
    if current_threshold is not None:
        if stat_type == 'max': thresh_str = f"_gt{int(current_threshold)}"
        if stat_type == 'min': thresh_str = f"_lt{int(current_threshold)}"

    save_result(result_df[final_cols], f"av_{file_prefix}_{stat_type}_temp.csv")


# ==========================================
# 道路方向熵计算
# ==========================================
def calculate_graph_entropy(G, num_bins=36):
    # 1. 获取边的属性
    bearings = []
    lengths = []

    if G is None or isinstance(G, bool):
        return 0

    try:
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'bearing' in data and 'length' in data:
                bearings.append(data['bearing'])
                lengths.append(data['length'])
    except AttributeError:
        return 0

    if not bearings:
        return 0

    bearings = np.array(bearings)
    bearings = (bearings + 360) % 360
    lengths = np.array(lengths)

    # 2. 制作直方图
    bins = np.linspace(0, 360, num_bins + 1)
    bin_indices = np.digitize(bearings, bins)

    # 3. 按桶累加长度
    bin_lengths = np.zeros(num_bins + 1)
    for idx, length in zip(bin_indices, lengths):
        if idx > num_bins:
            idx = num_bins
        bin_lengths[idx - 1] += length

    bin_lengths = bin_lengths[:num_bins]

    # 4. 计算概率分布 P
    total_length = bin_lengths.sum()
    if total_length == 0:
        return 0
    probs = bin_lengths / total_length

    # 5. 计算香农熵
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))

    return entropy


def process_single_file(file_info):
    """
    读取文件 -> 建图 -> 调用 calculate_graph_entropy
    """
    file_path, is_accessible, filename = file_info
    try:
        # 加载图
        G = ox.load_graphml(file_path)
        # 必须计算 bearing
        G = ox.bearing.add_edge_bearings(G)

        entropy = calculate_graph_entropy(G)

        # 格式化输出结果
        if not is_accessible:
            parts = filename.split(".")[0].split("_")
            if len(parts) >= 2:
                return [parts[0], parts[1], entropy]
            else:
                return [filename, "Unknown", entropy]
        else:
            return [filename.split(".")[0], entropy]

    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        return None


def compute_entropy(is_accessible):
    """
    这个函数接收 main 传来的 True/False (is_accessible)。
    负责遍历文件夹，分发任务。
    """
    sub_folder = "AV_Inaccessible" if not is_accessible else "AV_Accessible"
    input_dir = Path(PATHS["ENTROPY"]) / sub_folder

    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist.")
        return

    # 获取所有 .graphml 文件
    files = [f for f in os.listdir(input_dir) if f.endswith(".graphml")]
    tasks = []

    # 准备任务参数
    for f in files:
        full_path = input_dir / f
        tasks.append((full_path, is_accessible, f))

    print(f"Found {len(tasks)} files in {input_dir}")

    results = []
    # 并行处理
    if tasks:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks)))

    # 过滤掉出错的结果
    results = [r for r in results if r is not None]

    # 保存 CSV
    if not is_accessible:
        df = pd.DataFrame(results, columns=["ID_UC_G0", "GC_UCN_MAI_2025", "entropy"])
        output_filename = "av_inaccessible_entropy.csv"
    else:
        df = pd.DataFrame(results, columns=["city", "entropy"])
        output_filename = "av_accessible_entropy.csv"

    output_path = PATHS["RESULTS"] / output_filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Save to {output_path}")


# ==========================================
# 道路密度计算
# ==========================================
def calculate_road_length(G, unified_polygon, road_types, include_related_links=True):
    if not G or len(G.edges) == 0:
        return 0.0

    # 对于密度计算，我们需要物理长度。
    # 有向图(G)中，双向道路会有两条边(u->v, v->u)。
    # 转换为无向图会自动合并这两条边为一条，避免重复计算，也不需要后续除以2。
    G_undir = G.to_undirected()

    try:
        edges_gdf = ox.graph_to_gdfs(G_undir, nodes=False, edges=True)
    except Exception:
        return 0.0

    try:
        clipped_edges_gdf = gpd.clip(edges_gdf, unified_polygon)
    except Exception:
        return 0.0

    if clipped_edges_gdf.empty:
        return 0.0

    # 投影并计算长度
    utm_crs = clipped_edges_gdf.estimate_utm_crs()
    clipped_edges_proj = clipped_edges_gdf.to_crs(utm_crs)
    clipped_edges_proj['length'] = clipped_edges_proj.geometry.length

    if 'highway' in clipped_edges_proj.columns:
        is_list = clipped_edges_proj['highway'].apply(lambda x: isinstance(x, list))
        if is_list.any():
            clipped_edges_proj = clipped_edges_proj.explode('highway')

    # 因为已经是无向图，所有存在的边代表物理路段，
    # 其 length 属性就是实际物理长度，直接累加即可。
    lengths_by_type_m = clipped_edges_proj.groupby('highway', dropna=True)['length'].sum()
    lengths_by_type_km = (lengths_by_type_m / 1000).to_dict()

    road_length_by_type = {rt: 0.0 for rt in road_types}
    for rt in road_types:
        length = lengths_by_type_km.get(rt, 0.0)
        if include_related_links and not rt.endswith('_link'):
            length += lengths_by_type_km.get(f"{rt}_link", 0.0)
        road_length_by_type[rt] = length

    return sum(road_length_by_type.values())


def process_single_inaccessible_city(row_data, graph_dir, road_types):
    """
    处理单个 Inaccessible 城市：加载 GraphML -> 计算长度
    """
    city_id = str(row_data[ID_COL])
    city_name = row_data[NAME_COL]
    geometry = row_data['geometry']
    area_km2 = row_data["GC_UCA_KM2_2025"]

    # 构造 GraphML 路径
    graph_path = graph_dir / f"{city_id}_{city_name}.graphml"

    road_length_km = np.nan

    if graph_path.exists():
        try:
            G = ox.load_graphml(graph_path)
            road_length_km = calculate_road_length(
                G,
                unified_polygon=geometry,
                road_types=road_types,
                include_related_links=True
            )
        except Exception as e:
            pass

    return {
        ID_COL: city_id,
        NAME_COL: city_name,
        "road_length_km": road_length_km,
        "area_km2": area_km2
    }


def process_single_accessible_city(row_data, csv_dir):
    """
    处理单个 Accessible 城市：读取 CSV -> 提取 Total
    """
    city_name = row_data["city"]
    area_km2 = row_data["intersect_area_km2"]

    csv_filename = f"{city_name}_road_mileage.csv"
    csv_path = csv_dir / csv_filename

    road_length_km = np.nan

    if csv_path.exists():
        try:
            df_city_road = pd.read_csv(csv_path)
            total_row = df_city_road[df_city_road['road_type'] == 'total']
            if not total_row.empty:
                road_length_km = float(total_row.iloc[0]['value'])
        except Exception:
            pass

    return {
        "city": city_name,
        "road_length_km": road_length_km,
        "area_km2": area_km2
    }


def compute_road_density(is_accessible, max_workers=None):
    save_dir = PATHS["RESULTS"]
    os.makedirs(save_dir, exist_ok=True)

    results = []

    if not is_accessible:
        print("Processing AV-inaccessible cities (Parallel)...")
        gpkg_path = PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg"
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GPKG file not found: {gpkg_path}")

        gdf = gpd.read_file(gpkg_path)
        graph_dir = PATHS["ENTROPY"] / "AV_Inaccessible"

        # 准备数据列表，将 geometry 提取出来传给子进程
        # 注意：直接传递 row 会导致序列化开销，这里将其转换为字典
        tasks_data = []
        for _, row in gdf.iterrows():
            tasks_data.append({
                ID_COL: row[ID_COL],
                NAME_COL: row[NAME_COL],
                'geometry': row.geometry,
                "GC_UCA_KM2_2025": row["GC_UCA_KM2_2025"]
            })

        # 并行执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_inaccessible_city, data, graph_dir, ROAD_TYPES) for data in
                       tasks_data]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Density"):
                res = future.result()
                results.append(res)

        # 结果处理
        res_df = pd.DataFrame(results)
        res_df["road_density"] = res_df["road_length_km"] / res_df["area_km2"]
        final_res = res_df[[ID_COL, NAME_COL, "road_density"]].replace([np.inf, -np.inf], np.nan).dropna()

        out_path = save_dir / "av_inaccessible_road_density.csv"
        final_res.to_csv(out_path, index=False, encoding="utf-8")

    else:
        print("Processing AV-accessible cities (Parallel)...")
        if not os.path.exists(PATHS["AREA_CSV"]):
            print(f"Error: Area CSV not found at {PATHS['AREA_CSV']}")
            return

        df_area = pd.read_csv(PATHS["AREA_CSV"])
        csv_dir = PATHS["ACCESSIBLE_ROAD_DIR"]

        # 准备数据
        tasks_data = df_area.to_dict('records')  # 转换为字典列表

        # 并行执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_accessible_city, data, csv_dir) for data in tasks_data]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Reading CSVs"):
                res = future.result()
                results.append(res)

        # 结果处理
        df_res = pd.DataFrame(results)
        df_res["road_density"] = df_res["road_length_km"] / df_res["area_km2"]
        final_res = df_res[["city", "road_density"]].replace([np.inf, -np.inf], np.nan).dropna()

        out_path = save_dir / "av_accessible_road_density.csv"
        final_res.to_csv(out_path, index=False, encoding="utf-8")


# ==========================================
# 道路曲率 计算
# ==========================================
def process_single_sinuosity(file_info, is_accessible):
    """
    计算单个城市的 Sinuosity (蜿蜒度/曲率)
    Sinuosity = 路网实际长度 / 路网首尾直线距离
    在 OSMnx 中，basic_stats 的 'circuity_avg' 指标正是这个定义。
    """
    file_path, filename = file_info
    try:
        # 加载路网
        G = ox.load_graphml(file_path)

        # 计算基础统计量
        # circuity_avg = sum(edge_length) / sum(great_circle_distance)
        stats = ox.basic_stats(G)
        sinuosity = stats.get('circuity_avg')

        val = float(sinuosity) if sinuosity is not None else None

        # 处理文件名以获取 ID 和 Name
        if not is_accessible:
            parts = filename.split(".")[0].split("_")
            city_id = parts[0]
            city_name = "_".join(parts[1:]) if len(parts) > 1 else "Unknown"
            return [city_id, city_name, val]
        else:
            city_name = filename.split(".")[0]
            return [city_name, val]

    except Exception as e:
        print(f"Error processing sinuosity for {filename}: {e}")
        return None


def compute_sinuosity(is_accessible):
    """
    批量计算 Sinuosity 并保存 CSV
    """
    # 确定文件夹路径
    sub_folder = "AV_Inaccessible" if not is_accessible else "AV_Accessible"
    input_dir = Path(PATHS["ENTROPY"]) / sub_folder

    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist.")
        return

    # 获取所有 .graphml 文件
    files = [f for f in os.listdir(input_dir) if f.endswith(".graphml")]
    tasks = [(input_dir / f, f) for f in files]

    print(f"Calculating Sinuosity for {len(tasks)} files in {sub_folder}...")

    results = []
    # 并行计算
    with ProcessPoolExecutor() as executor:
        worker = partial(process_single_sinuosity, is_accessible=is_accessible)
        # 使用 tqdm 显示进度条
        results_iter = list(tqdm(executor.map(worker, tasks), total=len(tasks)))
        # 过滤掉 None 的结果
        results = [r for r in results_iter if r is not None]

    # 保存结果
    if not is_accessible:
        df = pd.DataFrame(results, columns=[ID_COL, NAME_COL, "sinuosity"])
        output_filename = "av_inaccessible_sinuosity.csv"
    else:
        df = pd.DataFrame(results, columns=["city", "sinuosity"])
        output_filename = "av_accessible_sinuosity.csv"

    # 调用你原本的保存函数
    save_result(df, output_filename)


# ==========================================
# 复杂交叉口比例计算
# ==========================================
def analyze_intersections(G, road_types=ROAD_TYPES, min_street_count=3):
    """
    统计包含环岛的总交叉口数量，并计算多臂路口数。

    - 总交叉口 = 普通有效交叉口（节点） + 环岛数量（每个计为1）
    - 普通交叉口：street_count >= min_street_count 且连接目标 road_types
    - 环岛上的节点被排除，避免重复计数
    - 多臂路口：普通交叉口中 street_count > 4 的节点
    """
    if not G or not getattr(G, 'nodes', None):
        return 0, 0, 0

    try:
        num_intersections, _, _, num_multi_arm, num_roundabouts = calculate_inter_number(G)
        return num_intersections, num_multi_arm, num_roundabouts
    except Exception as e:
        print(f"Error analyzing intersections: {e}")
        return 0, 0, 0


def process_single_ratio(file_info, is_accessible):
    file_path, filename = file_info
    try:
        G = ox.load_graphml(file_path)
        G = ox.convert.to_undirected(G)

        num_intersections, num_multi_arm, num_roundabouts = analyze_intersections(G)
        num_complex = num_multi_arm + num_roundabouts
        ratio_complex = (num_complex / num_intersections) * 100 if num_intersections > 0 else 0.0

        if not is_accessible:
            parts = filename.split(".")[0].split("_")
            city_id = parts[0]
            city = "_".join(parts[1:]) if len(parts) > 1 else "Unknown"
            return [
                city_id, city,
                num_intersections, num_multi_arm, num_roundabouts,
                num_complex, ratio_complex
            ]
        else:
            city_id = filename.split(".")[0]
            return [
                city_id,
                num_intersections, num_multi_arm, num_roundabouts,
                num_complex, ratio_complex
            ]

    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        return None


def compute_ratio_complex(is_accessible):
    if not is_accessible:
        sub_folder = "AV_Inaccessible"
        output_file = "av_inaccessible_ratio_complex.csv"
        cols = [
            "ID_UC_G0", "GC_UCN_MAI_2025", "num_intersections",
            "num_multi_arm", "num_roundabouts", "num_complex",
            "ratio_complex"
        ]
    else:
        sub_folder = "AV_Accessible"
        output_file = "av_accessible_ratio_complex.csv"
        cols = [
            "city", "num_intersections", "num_multi_arm",
            "num_roundabouts", "num_complex", "ratio_complex"
        ]

    directory = Path(PATHS["ENTROPY"]) / sub_folder
    if not directory.exists():
        print(f"Error: Directory not found {directory}")
        return

    file_list = [f for f in os.listdir(str(directory)) if f.endswith(".graphml")]
    tasks = [(directory / f, f) for f in file_list]
    print(f"Starting parallel processing for {len(tasks)} files in {sub_folder}...")

    results = []
    with ProcessPoolExecutor() as executor:
        worker = partial(process_single_ratio, is_accessible=is_accessible)
        results_iter = list(tqdm(executor.map(worker, tasks), total=len(tasks), unit="file"))
        results = [r for r in results_iter if r is not None]

    df = pd.DataFrame(results, columns=cols)
    output_path = PATHS["RESULTS"] / output_file
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Processing complete. Saved {len(df)} records to {output_path}")


# ==========================================
# POI丰度计算
# ==========================================
def calculate_richness_from_df(df, tags_keys):
    if df is None or df.empty:
        return 0, 0

    relevant_cols = [col for col in tags_keys if col in df.columns]

    if not relevant_cols:
        return len(df), 0

    # 使用 ravel('K') 展平所有标签列的值，计算唯一类型
    unique_types = pd.unique(df[relevant_cols].values.ravel('K'))
    # 过滤掉空值和空字符串
    unique_types = [t for t in unique_types if pd.notna(t) and str(t).strip() != '']

    return len(df), len(unique_types)


def download_pois_smart(geom, tags, identifier, chunk_threshold):
    # 1. 几何简化
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        simple_geom = geom.simplify(0.001, preserve_topology=True)
    else:
        simple_geom = geom

    # 2. 计算面积
    minx, miny, maxx, maxy = simple_geom.bounds
    area_sq_deg = (maxx - minx) * (maxy - miny)

    # ==========================================
    # 策略 A: 小区域直接下载
    # ==========================================
    if area_sq_deg < chunk_threshold:
        try:
            gdf = ox.features_from_polygon(simple_geom, tags=tags)
            if gdf.empty:
                return pd.DataFrame()

            # 立即轻量化处理
            gdf = gdf[gdf.geometry.type == 'Point']
            if gdf.empty:
                return pd.DataFrame()

            # 重置索引，将 osmid 变为普通列
            gdf = gdf.reset_index()
            return gdf
        except Exception as e:
            print(
                f"   ⚠️ Direct download failed for {identifier} (Area: {area_sq_deg:.4f}). Switching to chunked mode. Error: {e}")

    # ==========================================
    # 策略 B: 大区域切片下载
    # ==========================================
    grid_size = 0.05 if area_sq_deg < 1.0 else 0.1

    x_ranges = np.arange(minx, maxx, grid_size)
    y_ranges = np.arange(miny, maxy, grid_size)

    grid_boxes = []
    for x in x_ranges:
        for y in y_ranges:
            grid_box = box(x, y, x + grid_size, y + grid_size)
            if simple_geom.intersects(grid_box):
                try:
                    intersect_poly = simple_geom.intersection(grid_box)
                    if not intersect_poly.is_empty:
                        grid_boxes.append(intersect_poly)
                except ShapelyError:
                    # 如果几何计算出错，回退到使用矩形框
                    grid_boxes.append(grid_box)

    if not grid_boxes:
        return pd.DataFrame()

    print(f"   ▶️ Chunking {identifier}: Area {area_sq_deg:.2f} -> {len(grid_boxes)} chunks (Grid: {grid_size})")

    chunks = []

    # 定义单个切片下载函数 (线程内执行)
    def download_chunk(poly):
        try:
            # 下载
            chunk = ox.features_from_polygon(poly, tags=tags)
            if chunk.empty:
                return None

            # 1. 只保留点
            chunk = chunk[chunk.geometry.type == 'Point']
            if chunk.empty:
                return None

            # 2. 重置索引
            chunk = chunk.reset_index()

            # 3. 筛选列 (只保留需要的列，防止无关列占用内存)
            # 必须保留 osmid 用于去重，geometry 用于后续(虽然这里转字符串了)
            needed_cols = ['osmid', 'geometry'] + list(tags.keys())
            existing_cols = [c for c in needed_cols if c in chunk.columns]

            # 4. 转换为普通 DataFrame 并转为字符串
            # 这一步去除了 GeoDataFrame 的空间索引和 Shapely 对象，极大减少内存
            df_lite = pd.DataFrame(chunk[existing_cols]).astype(str)

            return df_lite

        except Exception:
            return None

    # 并行下载 -> 使用 ThreadPoolExecutor)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_chunk, poly) for poly in grid_boxes]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                chunks.append(result)

                # 内存保护：如果 chunks 列表过大，提前合并一次并触发 GC
                if len(chunks) > 50:
                    temp_df = pd.concat(chunks, ignore_index=True)
                    chunks = [temp_df]
                    gc.collect()

    if chunks:
        merged = pd.concat(chunks, ignore_index=True)
        # 去重 (因为切片边缘可能有重复点)
        if 'osmid' in merged.columns:
            merged = merged.drop_duplicates(subset=['osmid'])
        return merged
    else:
        return pd.DataFrame()


def process_single_poi(unit_data, is_accessible, chunk_threshold):
    identifier = unit_data.get('name', unit_data.get('city', 'Unknown_Unit'))

    try:
        poi_dir = PATHS["POI_DIR"]

        # ==========================================
        # 1. 准备路径、几何和标签
        # ==========================================
        if not is_accessible:
            # --- 模式 A: 不可达区域 ---
            fid = unit_data['fid']
            name = unit_data['name']
            geom = unit_data['geom']

            out_path = Path(poi_dir) / "AV_Inaccessible"
            out_path.mkdir(parents=True, exist_ok=True)

            # 文件名安全处理
            safe_name = re.sub(r'[^\w\-_.]', '_', str(name))[:50]
            csv_path = out_path / f"{fid}_{safe_name}.csv"
            tags_to_use = POI_TAGS
        else:
            # --- 模式 B: 可达区域 ---
            city_name = unit_data['city']
            geojson_path = unit_data['geojson_path']

            out_path = Path(poi_dir) / "AV_Accessible"
            out_path.mkdir(parents=True, exist_ok=True)
            csv_path = out_path / f"{city_name}.csv"

            if not os.path.exists(geojson_path):
                print(f"   ⚠️ GeoJSON not found: {geojson_path}")
                return None

            gdf_city = gpd.read_file(geojson_path)
            geom = gdf_city.unary_union

            tags_to_use = POI_TAGS.copy()
            # 特殊处理 Austin
            if city_name == "Austin":
                tags_to_use.pop('leisure', None)

        if geom is None or geom.is_empty:
            return None

        # ==========================================
        # 2. 缓存检查
        # ==========================================
        if csv_path.exists() and csv_path.stat().st_size > 10:
            try:
                # 读取缓存，全部作为字符串读取以加快速度
                df_cached = pd.read_csv(csv_path, dtype=str)
                poi_count, poi_richness = calculate_richness_from_df(df_cached, POI_TAGS.keys())

                result = {'poi_count': poi_count, 'poi_richness': poi_richness}
                if is_accessible:
                    result['city'] = city_name
                else:
                    result[ID_COL] = fid
                    result[NAME_COL] = name
                return result
            except Exception:
                pass  # 缓存读取失败，继续下载

        # ==========================================
        # 3. 智能下载
        # ==========================================
        print(f"⬇️ [Start] Downloading: {identifier}")

        # 调用优化后的下载函数，传入自定义阈值
        pois = download_pois_smart(geom, tags_to_use, identifier, chunk_threshold=chunk_threshold)

        print(f"✅ [End] {identifier}: {len(pois)} items")

        # ==========================================
        # 4. 处理数据与保存
        # ==========================================
        if not pois.empty:
            # 确保列存在
            key_cols = ['osmid'] + list(POI_TAGS.keys())
            existing_cols = [c for c in key_cols if c in pois.columns]

            pois_filtered = pois[existing_cols].copy()
            pois_filtered = pois_filtered.astype(str)

            # 保存
            pois_filtered.to_csv(csv_path, index=False, encoding="utf-8")

            # 计算
            poi_count, poi_richness = calculate_richness_from_df(pois_filtered, tags_to_use.keys())
        else:
            # 空结果处理，创建带表头的空 CSV
            empty_df = pd.DataFrame(columns=['osmid'] + list(POI_TAGS.keys()))
            empty_df.to_csv(csv_path, index=False, encoding="utf-8")
            poi_count, poi_richness = 0, 0

        # 手动清理内存
        del pois
        gc.collect()

        # ==========================================
        # 5. 返回结果
        # ==========================================
        result = {
            'poi_count': poi_count,
            'poi_richness': poi_richness
        }

        if is_accessible:
            result['city'] = city_name
        else:
            result[ID_COL] = fid
            result[NAME_COL] = name

        return result

    except Exception as e:
        print(f"❌ Critical Error processing {identifier}: {e}")
        # 出错时返回 None，主循环会过滤掉
        return None


def compute_poi_richness(is_accessible, chunk_threshold):
    if not is_accessible:
        # ========== 模式1: GPKG 多边形 ==========
        gdf = load_gpkg(PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg")
        tasks = []
        for _, row in gdf.iterrows():
            geom = row["geometry"]
            fid = row[ID_COL]
            name = row[NAME_COL]
            area_km2 = row.get("GC_UCA_KM2_2025", 0)

            if geom is None or geom.is_empty:
                continue

            tasks.append({
                'fid': fid,
                'name': name,
                'geom': geom,
                'area_km2': area_km2
            })
        output_file = "av_inaccessible_poi_richness.csv"
        columns = [ID_COL, NAME_COL, "poi_count", "poi_richness"]

    else:
        # ========== 模式2: 所有城市 ==========
        df_area = pd.read_csv(PATHS["AREA_CSV"])
        tasks = []
        for _, row in df_area.iterrows():
            city_name = row["city"]
            area_km2 = row["intersect_area_km2"]
            geojson_path = PATHS["GEOJSON"] / f"{city_name}.geojson"
            tasks.append({
                'city': city_name,
                'intersect_area_km2': area_km2,
                'geojson_path': str(geojson_path)
            })
        output_file = "av_accessible_poi_richness.csv"
        columns = ["city", "poi_count", "poi_richness"]

    # 并行处理
    # 使用 partial 将 chunk_threshold 传递给 process_single_poi
    worker = partial(process_single_poi, is_accessible=is_accessible, chunk_threshold=chunk_threshold)

    results = []
    # 根据机器性能调整 max_workers
    with ProcessPoolExecutor(max_workers=4) as executor:
        results_iter = tqdm(
            executor.map(worker, tasks),
            total=len(tasks),
            desc=f"POI Richness (Thresh: {chunk_threshold})",
            unit="city/polygon"
        )
        results = [r for r in results_iter if r is not None]

    # 汇总结果
    if results:
        df = pd.DataFrame(results)
        # Richness 是计数，不应该有 inf，但可以去除 NaN
        df = df.dropna(subset=["poi_richness"])
        save_result(df[columns], output_file)
        return df
    else:
        print("No valid results to save.")
        return pd.DataFrame()


# ==========================================
# POI类别熵计算
# ==========================================
def compute_poi_entropy(is_accessible: bool):
    # --- Helper to extract category ---
    def get_primary_category(row):
        for field in POI_TAGS:
            val = row.get(field)
            if pd.notna(val) and val != '':
                return val
        return None

    # --- Step 1: Prepare Filter Set (Consistency Check) ---
    valid_ids = set()

    if not is_accessible:
        # Mode 1: Inaccessible (Must filter based on GPKG)
        print("Loading GPKG for filtering...")
        try:
            gdf = load_gpkg(PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg")
            # Create a set of valid IDs for O(1) lookup
            valid_ids = set(gdf[ID_COL].astype(str).unique())
            print(f"Loaded {len(valid_ids)} valid IDs from GPKG.")
        except Exception as e:
            print(f"Error loading GPKG filter: {e}")
            return  # Critical failure if filter cannot be loaded
    else:
        # Mode 2: Accessible (Usually based on the Area CSV list, similar to richness function)
        # Although the prompt specifically mentioned the GPKG filter, it's good practice
        # to filter accessible cities too if strict consistency is required.
        # However, usually file existence is enough here unless strict alignment with df_area is needed.
        # We will proceed by processing all found CSVs, but you can add df_area filtering here if needed.
        pass

    results = []
    sub_folder = "AV_Inaccessible" if not is_accessible else "AV_Accessible"
    poi_dir = Path(PATHS["POI_DIR"]) / sub_folder

    # Get list of files to iterate with progress bar
    csv_files = list(poi_dir.glob("*.csv"))

    print(f"Computing POI Entropy for {sub_folder}...")

    for csv_file in tqdm(csv_files, desc="Calculating Entropy", unit="file"):
        file_stem = csv_file.stem

        # --- Step 2: Identification & Filtering ---
        current_id = None
        current_name = None

        if not is_accessible:
            # Expected format: "ID_Name.csv"
            parts = file_stem.split("_", 1)  # Split only on first underscore
            if len(parts) < 2:
                # Handle cases where filename might not match expected format
                current_id = parts[0]
                current_name = "Unknown"
            else:
                current_id = parts[0]
                current_name = parts[1]

            # *** CRITICAL FILTER ***
            # If this ID is not in our GPKG whitelist, skip it.
            if str(current_id) not in valid_ids:
                continue
        else:
            # Accessible format: "CityName.csv"
            current_name = file_stem
            # No ID filtering for accessible mode based on current requirements

        # --- Step 3: Load and Process Data ---
        try:
            # Read only necessary columns if possible to save memory,
            # but since we don't know exact col index of tags, we read all.
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Skipping {csv_file}: {e}")
            continue

        if df.empty:
            entropy = np.nan
            n_categories = 0
            total_poi = 0
        else:
            # Apply category extraction
            df['category'] = df.apply(get_primary_category, axis=1)
            df_valid = df.dropna(subset=['category'])

            if df_valid.empty:
                entropy = np.nan
                n_categories = 0
                total_poi = 0
            else:
                counts = df_valid['category'].value_counts()
                total_poi = counts.sum()
                probs = counts / total_poi
                # Shannon Entropy formula: -sum(p * log(p))
                entropy = -np.sum(probs * np.log(probs))
                n_categories = len(counts)

        # --- Step 4: Append Results ---
        if not is_accessible:
            results.append({
                ID_COL: current_id,
                NAME_COL: current_name,
                "poi_category_count": n_categories,
                "total_valid_poi": total_poi,
                "poi_entropy": entropy
            })
        else:
            results.append({
                "city": current_name,
                "poi_category_count": n_categories,
                "total_valid_poi": total_poi,
                "poi_entropy": entropy
            })

    # --- Step 5: Save Results ---
    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("No valid results computed.")
        return

    if not is_accessible:
        save_result(result_df, "av_inaccessible_poi_entropy.csv")
    else:
        save_result(result_df, "av_accessible_poi_entropy.csv")


# ==========================================
# 绘图
# ==========================================
def plot_box(indicator):
    # Hyperparameters of scienceplots
    plt.style.use(['science', 'no-latex', 'nature'])

    plt.rcParams.update({
        'font.size': 28,
        'axes.labelsize': 28,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 28,
        'legend.title_fontsize': 28,
        'lines.linewidth': 2,
        'axes.linewidth': 1.5,
        'xtick.direction': 'out',  # x轴刻度向外
        'ytick.direction': 'out',  # y轴刻度向外
        'xtick.major.size': 8,  # x轴主刻度线的长度
        'ytick.major.size': 8,  # y轴主刻度线的长度
        'xtick.major.width': 1.5,  # x轴主刻度线的宽度
        'ytick.major.width': 1.5,  # y轴主刻度线的宽度
        'xtick.top': False,  # 如果不需要顶部刻度可以设为False
        'ytick.right': False,  # 如果不需要右侧刻度可以设为False
        'font.family': 'sans-serif',
        'savefig.bbox': None
    })


    # 1. 数据读取与处理
    dfs = []
    for t in ["accessible", "inaccessible"]:
        df = pd.read_csv(PATHS["RESULTS"] / f"av_{t}_{indicator}.csv")
        val_cols = [c for c in df.columns if
                    indicator in c and c != "city" and c != "ID_UC_G0" and c != "GC_UCN_MAI_2025"]

        if len(val_cols) > 1:
            df[indicator] = df[val_cols].mean(axis=1)
        elif indicator not in df.columns and len(val_cols) == 1:
            df[indicator] = df[val_cols[0]]

        dfs.append(df[indicator].dropna().values)

    acc_vals, inacc_vals = dfs
    print(f"Indicator: {indicator}, Samples: Acc={len(acc_vals)}, Inacc={len(inacc_vals)}")

    # 2. 统计检验
    _, p = mannwhitneyu(acc_vals, inacc_vals, alternative='two-sided')
    p_text = "*** (P < 0.010)" if p < 0.01 else "** (P < 0.050)" if p < 0.05 else \
        "* (P < 0.100)" if p < 0.1 else "(P > 0.100)"

    # 3. 创建画布
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_axes([0.20, 0.12, 0.75, 0.82])
    ax.set_facecolor('#eaeaea')
    ax.grid(axis='y', linestyle='-', alpha=1, color='white', linewidth=1.5, zorder=0)

    # 定义颜色
    colors = ['#1f4e79', '#d66a54']  # 蓝 / 红

    # 3.5 绘制抖动散点
    jitter_strength = 0.08
    for i, vals in enumerate([acc_vals, inacc_vals]):
        x = np.random.normal(i + 1, jitter_strength, size=len(vals))
        ax.scatter(x, vals,
                   alpha=0.6,  # 透明度
                   s=120,  # 点的大小
                   color=colors[i],  # 使用对应颜色
                   edgecolor='none',
                   zorder=2)  # 层级在箱线图之下

    # 4. 绘制箱线图
    bplot = ax.boxplot([acc_vals, inacc_vals],
                       positions=[1, 2],
                       widths=0.5,
                       patch_artist=True,
                       showfliers=False,
                       notch=True,
                       zorder=3)

    # 5. 设置样式
    for patch in bplot['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        patch.set_alpha(1)

    # 设置须、帽、中位线为黑色
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bplot[element], color='black', linewidth=2.5)

    # 加粗中位线
    plt.setp(bplot['medians'], linewidth=2)

    # 5.5 绘制均值点
    means = [np.mean(acc_vals), np.mean(inacc_vals)]
    ax.scatter([1, 2], means, color='#333333', s=480, zorder=4, edgecolor='white', linewidth=2)

    # 6. 智能设置Y轴范围
    whisker_data = [item.get_ydata() for item in bplot['whiskers']]
    all_whisker_vals = [val for sublist in whisker_data for val in sublist]

    y_min_vis = min(all_whisker_vals)
    y_max_vis = max(all_whisker_vals)
    y_range_vis = y_max_vis - y_min_vis

    if y_range_vis == 0: y_range_vis = 1

    y_min_plot = y_min_vis - y_range_vis * 0.05
    if min(np.min(acc_vals), np.min(inacc_vals)) >= 0:
        y_min_plot = max(0, y_min_plot)

    y_max_plot = y_max_vis + y_range_vis * 0.40
    ax.set_ylim(y_min_plot, y_max_plot)

    # 7. 处理科学计数法
    sci_indicators = ['gdp', 'gdp_sum', 'pop_density', 'pop_size']
    forced_powers = {'gdp': 4}

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))

    if indicator in sci_indicators:
        if indicator in forced_powers:
            power = forced_powers[indicator]
            formatter = ticker.FuncFormatter(lambda x, pos: f'{x / 10 ** power:.1f}')
            ax.yaxis.set_major_formatter(formatter)
            offset_str = f'$\\times 10^{{{power}}}$'
        else:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
            fig.canvas.draw()
            offset_str = ax.yaxis.get_offset_text().get_text()

        if offset_str:
            ax.yaxis.offsetText.set_visible(False)
            ax.text(0, 1, offset_str, transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=22)

    # 8. 绘制显著性标记
    y_bottom, y_top = ax.get_ylim()
    y_span = y_top - y_bottom

    y_line = y_bottom + y_span * 0.90
    y_text = y_bottom + y_span * 0.92
    y_tips = y_bottom + y_span * 0.88

    ax.plot([1, 1, 2, 2], [y_tips, y_line, y_line, y_tips], color='black', lw=1.5)
    ax.text(1.5, y_text, p_text, ha='center', va='bottom', style='italic', color='black', fontweight='bold')

    # 9. 标签设置与边框清理
    ax.set_ylabel(LABELS.get(indicator, indicator), labelpad=10)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["AV-served", "AV-unserved"])
    ax.set_xlim(0.7, 2.3)

    # 10. 保存图片
    out_path = PATHS["RESULTS"] / f"{indicator}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==========================================
# 主程序
# ==========================================
def main(indicator="precipitation"):
    print(f"=== Analysis: {indicator} ===")

    filter_inaccessible_cities()

    if indicator == "precipitation":
        for f in os.listdir(PATHS["PRECIP"]):
            if f.startswith("precip.") and f.endswith(".nc"):
                out_name = f"precip_{f.split('.')[1]}.nc"
                if not (PATHS["PRECIP"] / out_name).exists():
                    convert_temp(PATHS["PRECIP"] / f, PATHS["PRECIP"] / out_name)
    elif indicator == "snowfall" and not (PATHS["SNOW"] / "snowfall.grib").exists():
        convert_snowfall(PATHS["SNOW"] / "ab5e5a91df06b88c6375d1e7035c5ade.grib", PATHS["SNOW"] / "snowfall.grib")
    elif "_temp" in indicator:
        for f in os.listdir(PATHS["TEMP"]):
            if (f.startswith("tmax.") or f.startswith("tmin.")) and f.endswith(".nc"):
                t_type = "max_temp" if "tmax" in f else "min_temp"
                out_name = f"{t_type}_{f.split('.')[1]}.nc"
                if not (PATHS["TEMP"] / out_name).exists():
                    convert_temp(PATHS["TEMP"] / f, PATHS["TEMP"] / out_name)
    elif indicator == "entropy":
        if os.path.exists(PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg"):
            process_gpkg_polygons(PATHS["FILTER"] / "GHS_UCDB_THEME_GHSL_GLOBE_R2024A_FILTER.gpkg",
                                  PATHS["ENTROPY"] / "AV_Inaccessible", ROAD_TYPES)

    tasks = {
        "gdp": lambda acc: compute_gdp(acc),
        "gdp_sum": lambda acc: compute_gdp_sum(acc),
        "pop_density": lambda acc: compute_pop_density(acc),
        "pop_size": lambda acc: compute_pop_size(acc),
        "annual_prep": lambda acc: compute_annual_prep(acc),
        "extreme_prep": lambda acc: compute_extreme_prep(acc, threshold_mm=20),
        "snowfall": lambda acc: compute_snowfall(acc),
        "max_temp": lambda acc: compute_temperature(acc, "max", 40),
        "min_temp": lambda acc: compute_temperature(acc, "min", -10),
        "slope": lambda acc: compute_raster_zonal_stat(PATHS["SLOPE_TIF"], acc, "slope"),
        "entropy": lambda acc: compute_entropy(acc),
        "road_density": lambda acc: compute_road_density(acc),
        "sinuosity": lambda acc: compute_sinuosity(acc),
        "ratio_complex": lambda acc: compute_ratio_complex(acc),
        "poi_richness": lambda acc: compute_poi_richness(acc, chunk_threshold=0.1),
        "poi_entropy": lambda acc: compute_poi_entropy(acc),
    }

    if indicator in tasks:
        print("--- Inaccessible ---")
        tasks[indicator](False)
        print("--- Accessible ---")
        tasks[indicator](True)
        print("--- Plotting ---")
        plot_box(indicator)


if __name__ == "__main__":
    for item in LABELS.keys():
        main(indicator=item)
