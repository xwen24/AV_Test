import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats


def calc_gdp_per_capita(POLY_GEOJSON,
                        RASTER_PATH="Data/QGIS/Gridded GDP per capita/rast_adm2_gdp_perCapita_1990_2022.tif",
                        TARGET_BAND=31):
    """
    基于栅格数据计算 GeoJSON/GPKG 多边形区域内的人均 GDP。
    返回: pandas.Series，**长度与原始文件行数一致**，无效几何或计算失败的位置为 NaN。
    """
    # 1. 读取多边形（支持 .geojson, .gpkg 等）
    polys = gpd.read_file(POLY_GEOJSON)

    # 2. 保留原始索引（不要 drop 行！）
    original_index = polys.index.copy()

    # 3. 处理 CRS
    if polys.crs is None:
        polys = polys.set_crs("EPSG:4326")

    with rasterio.open(RASTER_PATH) as src:
        raster_crs = src.crs or "EPSG:4326"
        raster_nodata = src.nodata
        if TARGET_BAND > src.count:
            raise ValueError(f"TARGET_BAND={TARGET_BAND} > total bands {src.count}")

    polys_projected = polys.to_crs(raster_crs)

    # 4. 标记哪些几何有效
    valid_geom_mask = (~polys.geometry.is_empty) & polys.geometry.notnull()
    # 创建一个全 NaN 的结果数组
    gdp_values = np.full(len(polys), np.nan)

    if valid_geom_mask.any():
        # 只对有效几何计算 zonal stats
        valid_polys = polys_projected[valid_geom_mask]

        nodata_to_use = raster_nodata if raster_nodata is not None else -9999

        stats = zonal_stats(
            vectors=valid_polys,
            raster=RASTER_PATH,
            band=TARGET_BAND,
            stats=['mean'],
            all_touched=True,
            nodata=nodata_to_use
        )

        # 提取 mean 值
        means = np.array([s['mean'] for s in stats], dtype=float)

        # 将结果放回原位置
        gdp_values[valid_geom_mask] = means

    # 返回与原始 polys 同索引的 Series
    return pd.Series(gdp_values, index=original_index, name='gdp_per_capita')