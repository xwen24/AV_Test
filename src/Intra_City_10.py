import os
import glob
import math
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box, shape, Point
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.plot import plotting_extent
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scienceplots
from scipy.stats import mannwhitneyu
from utilities.hyperparameters import ROAD_TYPES, CHN_CITY_NAME, POI_TAGS, POI_STATION_TAGS, LABELS, LABELS_UNITLESS
import networkx as nx
import osmnx as ox
ox.settings.use_cache = True
ox.settings.log_console = False
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import BallTree
import traceback


# ==========================================
# 配置区域
# ==========================================
GRID_RESOLUTION_METERS = 2000
ID_COL = "ID_UC_G0"
NAME_COL = "GC_UCN_MAI_2025"
LAYER_NAME = "GHS_UCDB_THEME_GHSL_GLOBE_R2024A"
ADMIN_SHP_PATH = 'Data/QGIS/ChinaAdminDivison/3. City/city.shp'

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "src/Results"

PATHS = {
    "GPKG": DATA_DIR / "QGIS/GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg",
    "GEOJSON": RESULTS_DIR / "Total_Dist/Built_Up",
    "RESULTS": RESULTS_DIR / f"Intra_City/Grid_{GRID_RESOLUTION_METERS}",
    "GRID_OUTPUT": RESULTS_DIR / f"Intra_City/Grid_{GRID_RESOLUTION_METERS}" / "grid",
    "OVERLAY_OUTPUT": RESULTS_DIR / f"Intra_City/Grid_{GRID_RESOLUTION_METERS}" / "overlay",
    "NTL_TIF": DATA_DIR / "QGIS/NTL/nppviirs_like_V2_2024.tif",
    "SLOPE_TIF": DATA_DIR / "QGIS/Slope/slope_1KMmn_SRTM.tif",
    "ENTROPY": BASE_DIR / "Data/OSM",
    "OSM_STATION": BASE_DIR / f"Data/OSM_Station/Grid_{GRID_RESOLUTION_METERS}",
    "POI_DIR": BASE_DIR / f"Data/POI_Detail/Grid_{GRID_RESOLUTION_METERS}"
}

for p in [PATHS["GRID_OUTPUT"], PATHS["RESULTS"]]:
    p.mkdir(parents=True, exist_ok=True)


# ==========================================
# 工具函数
# ==========================================
def clean_df(df):
    """清理DataFrame列名和乱码（BOM等）"""
    df.columns = df.columns.str.replace(r'^\ufeff', '', regex=True).str.strip()
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('\ufeff', '', regex=False)
            .str.replace(r'^Ôªø|^ï»¿', '', regex=True)
            .replace('nan', pd.NA)
        )
    return df


def load_gpkg(path, layer=None):
    return clean_df(gpd.read_file(path, layer=layer))


def load_china_admin():
    if not os.path.exists(ADMIN_SHP_PATH):
        print(f"⚠️ 警告: 中国行政边界文件未找到: {ADMIN_SHP_PATH}")
        return None
    print("加载中国市级行政边界...")
    china_gdf = gpd.read_file(ADMIN_SHP_PATH)
    if china_gdf.crs != "EPSG:4326":
        china_gdf = china_gdf.to_crs("EPSG:4326")
    return china_gdf


def create_square_grid(boundary_gdf, resolution, city_id_prefix="UNKNOWN", target_crs=None):
    if target_crs is None:
        target_crs = boundary_gdf.estimate_utm_crs()

    projected_boundary = boundary_gdf.to_crs(target_crs)
    xmin, ymin, xmax, ymax = projected_boundary.total_bounds

    xs = np.arange(xmin, xmax, resolution)
    ys = np.arange(ymin, ymax, resolution)

    polygons = []
    for x in xs:
        for y in ys:
            poly = box(x, y, x + resolution, y + resolution)
            polygons.append(poly)

    grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs)
    mask = projected_boundary.unary_union
    grid_gdf = grid_gdf[grid_gdf.intersects(mask)].copy()
    grid_gdf['grid_id'] = [f"{city_id_prefix}_{i}" for i in range(len(grid_gdf))]
    return grid_gdf.to_crs("EPSG:4326")


def process_single_grid_file(geojson_path, global_cities_gdf, china_admin_gdf=None):
    file_name = geojson_path.name
    print(f"\n--- [Grid Gen] 处理文件: {file_name} ---")

    try:
        dep_gdf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return None

    if dep_gdf.crs != "EPSG:4326":
        dep_gdf = dep_gdf.to_crs("EPSG:4326")

    dep_union = dep_gdf.unary_union
    dep_union_gdf = gpd.GeoDataFrame({'geometry': [dep_union]}, crs="EPSG:4326")

    # 从文件名提取城市名
    std_city_name = file_name.split('-')[0].split('.')[0]
    is_china_city = std_city_name in CHN_CITY_NAME

    # === 第一步：始终用 global_cities_gdf 匹配，获取标准城市 ID 和名称 ===
    matched_cities = gpd.sjoin(global_cities_gdf, dep_union_gdf, how="inner", predicate="intersects")
    if matched_cities.empty:
        print(f"⚠️ {file_name} 没有匹配到任何城市（全球数据），跳过。")
        return None

    matched_names = matched_cities[NAME_COL].unique().tolist()
    matched_ids = matched_cities[ID_COL].unique().tolist()
    main_city_id = matched_ids[0]
    print(f"✅ 全球城市匹配: {matched_names} (ID: {main_city_id})")

    # 获取全球数据定义的城市几何（Built-up Area）
    global_city_geom = matched_cities.geometry.unary_union

    # === 第二步：确定用于生成网格的“背景几何” ===
    if is_china_city and china_admin_gdf is not None:
        # 获取对应的中国行政边界
        admin_name = CHN_CITY_NAME[std_city_name]
        admin_polygons = china_admin_gdf[china_admin_gdf['ct_name'] == admin_name]

        if not admin_polygons.empty:
            admin_geom = admin_polygons.unary_union

            # 使用行政边界 裁剪 全球城市建成区
            # Intersection: 取 全球建成区 和 行政边界 的重叠部分
            city_geom_for_grid = global_city_geom.intersection(admin_geom)

            # 完整性检查：如果裁剪后为空（极其罕见，可能是坐标系偏差或数据不匹配），回退到全球数据
            if city_geom_for_grid.is_empty:
                print(f"⚠️ 警告: {admin_name} 行政边界与全球城市数据无交集，回退使用全球城市几何。")
                city_geom_for_grid = global_city_geom
            else:
                print(f"✅ 中国城市处理: 已使用行政边界 '{admin_name}' 裁剪全球城市建成区")
        else:
            # 未找到行政边界，回退
            city_geom_for_grid = global_city_geom
            print(f"⚠️ 未找到中国行政边界 '{admin_name}'，使用全球城市几何")
    else:
        # 非中国城市：直接用全球匹配结果
        city_geom_for_grid = global_city_geom

    # === 第三步：合并部署区域 ===
    # 背景 = (裁剪后的城市建成区) ∪ (部署区域)
    # 这一步确保即使行政边界裁剪切掉了一部分，部署区域本身依然会被网格覆盖
    deployment_geom = dep_union_gdf.geometry.iloc[0]
    combined_geom = city_geom_for_grid.union(deployment_geom).buffer(0)
    background_gdf = gpd.GeoDataFrame({'geometry': [combined_geom]}, crs="EPSG:4326")

    # === 生成网格 ===
    print(f"正在生成正方形网格 (分辨率: {GRID_RESOLUTION_METERS}米)...")
    target_utm_crs = dep_union_gdf.estimate_utm_crs()
    grid_gdf = create_square_grid(
        background_gdf,
        resolution=GRID_RESOLUTION_METERS,
        city_id_prefix=main_city_id,
        target_crs=target_utm_crs
    )

    # === 判定部署状态（>50% 覆盖）===
    grid_utm = grid_gdf.to_crs(target_utm_crs)
    dep_utm_geom = dep_union_gdf.to_crs(target_utm_crs).geometry.iloc[0]
    grid_areas = grid_utm.geometry.area
    intersection_areas = grid_utm.geometry.intersection(dep_utm_geom).area
    overlap_ratios = intersection_areas / grid_areas
    grid_gdf['is_deployed'] = (overlap_ratios > 0.5).astype(int)

    # === 添加元数据（来自全球城市匹配）===
    grid_gdf['city_id'] = ",".join(map(str, matched_ids))
    grid_gdf['city_name'] = ",".join(map(str, matched_names))
    grid_gdf['source_file'] = file_name

    deployed_count = grid_gdf['is_deployed'].sum()
    print(f"统计: 总格子 {len(grid_gdf)}, 已部署 {deployed_count}")

    return grid_gdf


def run_grid_generation(acc=None):
    print("\n>>> 开始执行网格生成任务 <<<")

    # 创建输出目录
    PATHS['GRID_OUTPUT'].mkdir(parents=True, exist_ok=True)

    # 加载全球城市数据
    if not os.path.exists(PATHS['GPKG']):
        print(f"❌ 错误: 找不到 GPKG 文件 {PATHS['GPKG']}")
        return
    print("🌍 正在加载全球城市数据 (GPKG)...")
    global_cities = load_gpkg(PATHS['GPKG'], layer=LAYER_NAME)
    if global_cities.crs != "EPSG:4326":
        global_cities = global_cities.to_crs("EPSG:4326")

    # 加载中国行政边界
    china_admin_gdf = load_china_admin()

    # 处理部署区域文件
    if not PATHS['GEOJSON'].exists():
        print(f"❌ 错误: 找不到部署区域输入目录 {PATHS['GEOJSON']}")
        return

    all_files = list(PATHS['GEOJSON'].glob("*.geojson"))
    print(f"📁 找到 {len(all_files)} 个部署区域文件待处理。")

    for f in all_files:
        output_filename = f"{f.stem}.geojson"
        output_path = PATHS['GRID_OUTPUT'] / output_filename

        if output_path.exists():
            print(f"⏭️ 跳过已存在的网格文件: {output_path}")
            continue

        result_gdf = process_single_grid_file(f, global_cities, china_admin_gdf)
        if result_gdf is not None:
            result_gdf.to_file(output_path, driver="GeoJSON")
            print(f"💾 已保存网格文件: {output_path}")


def compute_raster_mean(tif_path, indicator_name):
    print(f"\n>>> 开始计算栅格均值 [{indicator_name}]: {tif_path.name} <<<")

    if not tif_path.exists():
        print(f"错误: 找不到栅格文件 {tif_path}")
        return

    grid_files = list(PATHS['GRID_OUTPUT'].glob("*.geojson"))
    print(f"待处理网格文件数量: {len(grid_files)}")

    for grid_file in grid_files:
        csv_filename = PATHS["RESULTS"] / indicator_name / f"{grid_file.stem}.csv"

        if csv_filename.exists():
            print(f"  [跳过] 文件已存在: {csv_filename}")
            continue

        try:
            print(f"正在处理: {grid_file.name} ...")

            grid_gdf = gpd.read_file(grid_file)
            if grid_gdf.empty:
                print("  警告: 网格文件为空，跳过。")
                continue

            # 准备结果列表
            results = []

            with rasterio.open(tif_path) as src:
                # 1. 关键步骤：必须将网格重投影到栅格的 CRS，否则 mask 会失效
                if grid_gdf.crs != src.crs:
                    grid_gdf_proj = grid_gdf.to_crs(src.crs)
                else:
                    grid_gdf_proj = grid_gdf

                nodata_val = src.nodata

                # 2. 遍历每个网格单元进行统计
                # 使用 iterrows 可能会稍慢，但对于几千个网格是可以接受的，且逻辑最清晰
                for idx, row in grid_gdf_proj.iterrows():
                    geom = row.geometry
                    grid_id = row['grid_id']

                    try:
                        # 3. Mask: 提取几何体范围内的像素
                        # all_touched=True: 只要像素与网格有接触就纳入计算（防止网格极小落入像素缝隙）
                        # crop=True: 裁剪输出数组的大小以适应几何体
                        out_image, out_transform = mask(src, [geom], crop=True, all_touched=True)

                        # out_image 维度是 (band_count, height, width)，我们取第一个波段
                        data = out_image[0]

                        # 4. 数据清洗
                        # 将 nodata 值替换为 NaN 以便计算平均值
                        if nodata_val is not None:
                            data = data.astype('float32')  # 确保可以存放 NaN
                            data[data == nodata_val] = np.nan

                        # 还可以过滤掉异常值（例如坡度不可能小于0）
                        data[data < 0] = np.nan

                        # 5. 计算平均值
                        # 使用 nanmean 忽略 NaN 值
                        # 如果全是 NaN (例如网格在海上或无数据区)，结果会是 NaN
                        if np.isnan(data).all():
                            mean_val = 0.0  # 或者 np.nan，视你的统计需求而定
                        else:
                            mean_val = np.nanmean(data)

                    except ValueError:
                        # 通常发生在几何体不在栅格范围内
                        mean_val = 0.0
                    except Exception as e:
                        print(f"    计算 Grid {grid_id} 出错: {e}")
                        mean_val = 0.0

                    results.append({
                        'grid_id': grid_id,
                        indicator_name: round(float(mean_val), 4)  # 保留4位小数
                    })

            # 6. 整理输出
            result_df = pd.DataFrame(results)

            # 合并回原始信息（如果需要 is_deployed 状态）
            # 这里我们只保留 grid_id 和 计算结果，保持 CSV 轻量
            # 如果需要 is_deployed，可以从原始 grid_gdf merge 过来
            if 'is_deployed' in grid_gdf.columns:
                meta_df = grid_gdf[['grid_id', 'is_deployed']].copy()
                meta_df['is_deployed'] = meta_df['is_deployed'].astype(int)
                output_df = meta_df.merge(result_df, on='grid_id', how='left')
            else:
                output_df = result_df

            # 7. 保存 CSV
            output_path = PATHS["RESULTS"] / indicator_name / f"{grid_file.stem}.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"  处理 {grid_file.name} 时发生错误: {e}")
            traceback.print_exc()


# ==========================================
# 夜间灯光强度
# ==========================================
def get_raster_pixels_as_polygons(src, bounds, crs):
    minx, miny, maxx, maxy = bounds

    # 1. 计算窗口
    try:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
    except Exception:
        return None

    # 2. 读取数据
    data = src.read(1, window=window)
    transform = src.window_transform(window)

    # 处理 nodata
    nodata = src.nodata
    if nodata is None:
        nodata = -9999

    # 创建掩膜：只处理有效数据
    mask = (data != nodata) & (~np.isnan(data))

    if not mask.any():
        return None

    # 3. 快速生成几何体
    results = (
        {'properties': {'pixel_value': v}, 'geometry': shape(s)}
        for s, v in shapes(data, mask=mask, transform=transform)
    )

    # 4. 创建 GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)
    return gdf


def compute_ntl_mean(tif_path, indicator_name):
    print(f"\n>>> 开始计算夜间灯光强度: {tif_path.name} <<<")

    if not tif_path.exists():
        print(f"错误: 找不到夜间灯光栅格文件 {tif_path}")
        return

    # --- 路径设置 ---
    # 1. 原始输出目录 (比如 ntl_mean)
    output_dir_origin = PATHS["RESULTS"] / indicator_name
    os.makedirs(output_dir_origin, exist_ok=True)

    # 2. 对数输出目录 (固定为 ntl_mean_log)
    output_dir_log = PATHS["RESULTS"] / "ntl_mean_log"
    os.makedirs(output_dir_log, exist_ok=True)

    grid_files = list(PATHS['GRID_OUTPUT'].glob("*.geojson"))
    print(f"待处理网格文件数量: {len(grid_files)}")

    for grid_file in grid_files:
        # 定义两个输出文件的路径
        csv_origin = output_dir_origin / f"{grid_file.stem}.csv"
        csv_log = output_dir_log / f"{grid_file.stem}.csv"

        # 如果两个文件都存在，则跳过
        if csv_origin.exists() and csv_log.exists():
            print(f"  [跳过] 文件已存在: {grid_file.stem}")
            continue

        try:
            print(f"正在处理: {grid_file.name} ...")

            grid_gdf = gpd.read_file(grid_file)
            if grid_gdf.empty:
                print("  警告: 网格文件为空，跳过。")
                continue

            # 1. 投影转换与面积计算
            utm_crs = grid_gdf.estimate_utm_crs()
            grid_utm = grid_gdf.to_crs(utm_crs)
            grid_utm['grid_area_m2'] = grid_utm.geometry.area

            # 2. 读取夜光栅格数据
            with rasterio.open(tif_path) as src:
                if grid_gdf.crs != src.crs:
                    grid_gdf_for_query = grid_gdf.to_crs(src.crs)
                else:
                    grid_gdf_for_query = grid_gdf

                bounds = grid_gdf_for_query.total_bounds
                pixel_gdf = get_raster_pixels_as_polygons(src, bounds, src.crs)

                if pixel_gdf is None:
                    print("  警告: 该区域无有效夜光数据。")
                    grid_utm['ntl_mean'] = 0.0
                    grid_utm['ntl_sum'] = 0.0
                else:
                    # 4. 统一投影到 UTM 进行几何运算
                    pixel_gdf_utm = pixel_gdf.to_crs(utm_crs)

                    # 5. 空间叠加
                    intersection = gpd.overlay(
                        grid_utm[['grid_id', 'geometry']],
                        pixel_gdf_utm[['pixel_value', 'geometry']],
                        how='intersection'
                    )

                    if intersection.empty:
                        grid_utm['ntl_mean'] = 0.0
                        grid_utm['ntl_sum'] = 0.0
                    else:
                        # 6. 计算权重
                        intersection['intersect_area'] = intersection.geometry.area
                        intersection['weighted_val'] = intersection['pixel_value'] * intersection['intersect_area']

                        # 7. 汇总回网格 ID
                        stats = intersection.groupby('grid_id').agg({
                            'weighted_val': 'sum',
                            'intersect_area': 'sum'
                        }).reset_index()

                        grid_utm = grid_utm.merge(stats, on='grid_id', how='left')
                        grid_utm['weighted_val'] = grid_utm['weighted_val'].fillna(0)

                        # 8. 计算指标
                        grid_utm['ntl_mean'] = grid_utm['weighted_val'] / grid_utm['grid_area_m2']
                        grid_utm['ntl_sum'] = grid_utm['weighted_val']

            # 9. 整理基础数据
            if 'is_deployed' not in grid_utm.columns:
                grid_utm['is_deployed'] = 0

            grid_utm['is_deployed'] = grid_utm['is_deployed'].astype(int)

            # --- 输出 1: ntl_mean ---
            if not csv_origin.exists():
                cols_origin = ['grid_id', 'is_deployed', 'ntl_mean', 'ntl_sum']
                df_origin = grid_utm[cols_origin].copy()
                df_origin['ntl_mean'] = df_origin['ntl_mean'].round(4)
                df_origin['ntl_sum'] = df_origin['ntl_sum'].round(2)
                df_origin.to_csv(csv_origin, index=False, encoding='utf-8-sig')

            # --- 输出 2: ntl_mean_log ---
            if not csv_log.exists():
                grid_utm['ntl_mean_log'] = np.log1p(grid_utm['ntl_mean'])
                cols_log = ['grid_id', 'is_deployed', 'ntl_mean', 'ntl_mean_log']
                df_log = grid_utm[cols_log].copy()
                df_log['ntl_mean'] = df_log['ntl_mean'].round(4)
                df_log['ntl_mean_log'] = df_log['ntl_mean_log'].round(4)
                df_log.to_csv(csv_log, index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"  处理 {grid_file.name} 时发生错误: {e}")
            traceback.print_exc()


# ==========================================
# 道路方向熵计算
# ==========================================
def calculate_entropy_numpy(bearings, lengths, num_bins=36):
    valid = np.isfinite(bearings) & np.isfinite(lengths)
    bearings = bearings[valid]
    lengths = lengths[valid]

    if len(bearings) == 0 or len(lengths) == 0:
        return 0.0

    # 确保 bearings 在 0-360 之间
    bearings = (bearings + 360) % 360

    # 制作直方图 (Binning)
    # bins: 0, 10, 20, ..., 360
    bins = np.linspace(0, 360, num_bins + 1)

    # 获取每个 bearing 属于哪个桶 (1 到 num_bins)
    bin_indices = np.digitize(bearings, bins)

    # 处理边界 (360度可能归到最后一个桶溢出，归入第36桶)
    bin_indices[bin_indices > num_bins] = num_bins

    # 按桶累加长度 (Weighted Sum)
    # bin_indices - 1 是为了让索引从 0 开始
    # minlength 确保即使某些方向没有路，桶也是存在的(值为0)
    bin_lengths = np.bincount(bin_indices - 1, weights=lengths, minlength=num_bins)

    # 只取前 num_bins 个 (防止 digitize 产生的溢出)
    bin_lengths = bin_lengths[:num_bins]

    # 计算概率分布 P
    total_length = bin_lengths.sum()
    if total_length == 0:
        return 0.0

    probs = bin_lengths / total_length

    # 计算香农熵 (忽略概率为0的项，因为 log(0) 无定义)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))

    return entropy


def compute_geometry_bearing(gdf):
    def get_coords(geom):
        if geom is None or geom.is_empty:
            return np.nan, np.nan, np.nan, np.nan
        if geom.geom_type == 'LineString':
            coords = geom.coords
            if len(coords) < 2:
                return np.nan, np.nan, np.nan, np.nan
            return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]
        elif geom.geom_type == 'MultiLineString':
            if len(geom.geoms) == 0:
                return np.nan, np.nan, np.nan, np.nan
            # 取最长的一段
            longest = max(geom.geoms, key=lambda g: g.length)
            coords = longest.coords
            if len(coords) < 2:
                return np.nan, np.nan, np.nan, np.nan
            return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]
        else:
            return np.nan, np.nan, np.nan, np.nan

    coords = gdf.geometry.apply(get_coords).tolist()
    coords_df = pd.DataFrame(coords, columns=['x1', 'y1', 'x2', 'y2'], index=gdf.index)

    dy = coords_df['y2'] - coords_df['y1']
    dx = coords_df['x2'] - coords_df['x1']
    bearings = np.degrees(np.arctan2(dx, dy))
    bearings = (bearings + 360) % 360
    return bearings


def process_single_road_entropy(city_name, grid_file, osm_dir, indicator_name="road_entropy"):
    output_csv = PATHS["RESULTS"] / indicator_name / f"{city_name}.csv"
    if output_csv.exists():
        print(f"  [跳过] {city_name}.csv 已存在")
        return

    # 1. 读取网格
    grid_gdf = gpd.read_file(grid_file)
    if grid_gdf.empty:
        return

    # 2. 加载路网
    try:
        edges_gdf = merge_city_road_graph(osm_dir, city_name)
    except Exception as e:
        print(f"  ❌ 加载路网失败 {city_name}: {e}")
        return

    if edges_gdf.empty:
        print(f"  ⚠️ {city_name} 路网为空，生成零值结果。")
        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['entropy'] = 0.0
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        return

    # 3. 统一投影到 UTM (计算长度和角度必须在投影坐标系下)
    utm_crs = grid_gdf.estimate_utm_crs()
    grid_utm = grid_gdf.to_crs(utm_crs)
    edges_utm = edges_gdf.to_crs(utm_crs)

    # 4. 空间裁剪 (Overlay)
    print(f"  - 正在计算 {city_name} 的路网熵 (Vectorized Overlay)...")

    grid_subset = grid_utm[['grid_id', 'geometry']]
    edges_subset = edges_utm[['geometry']]

    try:
        # Intersection: 将路网按网格切断
        intersections = gpd.overlay(
            edges_subset,
            grid_subset,
            how='intersection',
            keep_geom_type=False
        )
    except Exception as e:
        print(f"  ❌ Overlay 计算失败: {e}")
        return

    if intersections.empty:
        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['entropy'] = 0.0
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        return

    # 5. 计算属性 (在切断后的路段上计算)
    # A. 计算长度
    intersections['seg_len'] = intersections.geometry.length

    # B. 计算方向 (Bearing) - 关键步骤
    intersections['seg_bearing'] = compute_geometry_bearing(intersections)

    # 6. 分组计算熵 (GroupBy Apply)
    print(f"  - 正在聚合计算熵...")

    # 定义聚合逻辑：对每个 grid_id 组，取出 bearing 和 len 数组传给 numpy 函数
    def agg_entropy(df):
        return calculate_entropy_numpy(
            df['seg_bearing'].values,
            df['seg_len'].values
        )

    # groupby().apply() 可能会稍慢，但对于几万个网格通常在可接受范围内
    entropy_stats = intersections.groupby('grid_id').apply(agg_entropy).reset_index(name='entropy')

    # 7. 合并回原始网格
    final_df = grid_utm.merge(entropy_stats, on='grid_id', how='left')

    # 填充 NaN (没有路网的格子熵为 0)
    final_df['entropy'] = final_df['entropy'].fillna(0)
    final_df['entropy'] = final_df['entropy'].round(4)

    # 8. 保存
    output_df = final_df[['grid_id', 'is_deployed', 'entropy']]
    os.makedirs(output_csv.parent, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def compute_road_entropy(indicator_name="entropy"):
    print("\n>>> 开始按城市计算道路方向熵 (Grid-level) <<<")

    # 确保输出目录存在
    os.makedirs(PATHS["RESULTS"] / indicator_name, exist_ok=True)

    # 获取所有城市名
    city_dirs = [d for d in os.listdir(PATHS["GEOJSON"]) if d.endswith(".geojson")]
    print(f"📍 共识别到 {len(city_dirs)} 个城市")

    for city_dir in sorted(city_dirs):
        city_name = city_dir.split(".")[0]
        grid_file = PATHS['GRID_OUTPUT'] / f"{city_name}.geojson"

        if not grid_file.exists():
            print(f"⚠️  跳过 {city_name}：缺少网格文件")
            continue

        try:
            process_single_road_entropy(
                city_name=city_name,
                grid_file=grid_file,
                osm_dir=PATHS["ENTROPY"],
                indicator_name=indicator_name
            )
        except Exception as e:
            print(f"❌ 处理城市 {city_name} 时出错: {e}")
            traceback.print_exc()


# ==========================================
# 道路密度
# ==========================================
def merge_city_road_graph(city_road_dir, city_name):
    """
    加载、合并并提取无向图的边（物理路段）。
    """
    accessible = city_road_dir / "AV_Accessible" / f"{city_name}.graphml"
    inaccessible = city_road_dir / "AV_Inaccessible_Uncovered" / f"{city_name}.graphml"

    if not accessible.exists():
        raise FileNotFoundError(f"缺少文件: {accessible}")
    if not inaccessible.exists():
        raise FileNotFoundError(f"缺少文件: {inaccessible}")

    print(f"  - 加载 {city_name} 的路网...")
    G1 = ox.load_graphml(accessible)
    G2 = ox.load_graphml(inaccessible)

    # 获取原始 CRS
    original_crs = G1.graph.get('crs', "epsg:4326")

    # 合并图
    G = nx.compose(G1, G2)

    # --- 关键步骤：转无向图 ---
    # 这确保了双向车道合并为一条物理边，计算的是物理长度
    G_undir = G.to_undirected()

    # 提取边 GeoDataFrame
    edges = ox.graph_to_gdfs(G_undir, nodes=False, edges=True)

    # 恢复 CRS
    if edges.crs is None:
        edges = edges.set_crs(original_crs, allow_override=True)

    # 统一输出为 4326
    if edges.crs != "epsg:4326":
        edges = edges.to_crs("epsg:4326")

    return edges


def process_single_road_density(city_name, grid_file, osm_dir, indicator_name="road_density"):
    output_csv = PATHS["RESULTS"] / indicator_name / f"{city_name}.csv"
    if output_csv.exists():
        print(f"  [跳过] {city_name}.csv 已存在")
        return

    # 1. 读取网格
    grid_gdf = gpd.read_file(grid_file)
    if grid_gdf.empty:
        return

    # 2. 加载路网
    try:
        edges_gdf = merge_city_road_graph(osm_dir, city_name)
    except Exception as e:
        print(f"  ❌ 加载路网失败 {city_name}: {e}")
        return

    if edges_gdf.empty:
        print(f"  ⚠️ {city_name} 路网为空，生成零值结果。")
        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['road_density'] = 0.0
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        return

    # 3. 统一投影到 UTM (计算长度和面积必须在投影坐标系下)
    # 使用网格的中心来估算最佳 UTM 投影带
    utm_crs = grid_gdf.estimate_utm_crs()
    grid_utm = grid_gdf.to_crs(utm_crs)
    edges_utm = edges_gdf.to_crs(utm_crs)

    # 计算网格面积 (km2)
    grid_utm['grid_area_km2'] = grid_utm.geometry.area / 1e6

    # 4. 核心计算：使用 Overlay 进行批量裁剪
    print(f"  - 正在计算 {city_name} 的路网密度 (Vectorized Overlay)...")

    # 为了加快速度，只保留必要的列进行几何运算
    # grid 只留 id 和 geometry
    grid_subset = grid_utm[['grid_id', 'geometry']]
    # edges 只留 geometry (因为不需要筛选类型了)
    edges_subset = edges_utm[['geometry']]

    try:
        # Overlay: Intersection
        # 这会将路网按照网格边界切断。
        # 结果中的每一行都是一段路，且带有一个 grid_id。
        intersections = gpd.overlay(
            edges_subset,
            grid_subset,
            how='intersection',
            keep_geom_type=False  # 允许线被切成点（虽然计算长度时点为0，不影响）
        )
    except Exception as e:
        print(f"  ❌ Overlay 计算失败: {e}")
        return

    # 5. 计算切断后的长度
    # 长度单位转为 km
    intersections['seg_len_km'] = intersections.geometry.length / 1000.0

    # 6. 聚合统计 (Groupby grid_id)
    # 计算每个 grid 内的总路长
    stats = intersections.groupby('grid_id')['seg_len_km'].sum().reset_index()
    stats.rename(columns={'seg_len_km': 'total_road_len_km'}, inplace=True)

    # 7. 合并回原始网格数据
    # 使用 left join 确保保留那些没有路网的格子（它们的值将是 NaN）
    final_df = grid_utm.merge(stats, on='grid_id', how='left')

    # 填充 NaN 为 0
    final_df['total_road_len_km'] = final_df['total_road_len_km'].fillna(0)

    # 计算密度: km / km2
    final_df['road_density'] = final_df['total_road_len_km'] / final_df['grid_area_km2']

    # 处理除以零的异常（极少数情况网格面积可能为0）
    final_df.loc[final_df['grid_area_km2'] == 0, 'road_density'] = 0

    # 四舍五入
    final_df['road_density'] = final_df['road_density'].round(4)

    # 8. 保存结果
    output_df = final_df[['grid_id', 'is_deployed', 'road_density']]
    os.makedirs(output_csv.parent, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def compute_road_density(indicator_name="road_density"):
    print("\n>>> 开始按城市计算道路密度 <<<")

    # 获取所有城市名
    city_dirs = [d for d in os.listdir(PATHS["GEOJSON"]) if d.endswith(".geojson")]
    print(f"📍 共识别到 {len(city_dirs)} 个城市: {sorted(city_dirs)}")

    for city_dir in sorted(city_dirs):
        city_name = city_dir.split(".")[0]
        grid_file = PATHS['GRID_OUTPUT'] / f"{city_name}.geojson"

        if not grid_file.exists():
            print(f"⚠️  跳过 {city_name}：缺少网格文件 {grid_file}")
            continue

        try:
            process_single_road_density(
                city_name=city_name,
                grid_file=grid_file,
                osm_dir=PATHS["ENTROPY"],
                indicator_name=indicator_name
            )
        except Exception as e:
            print(f"❌ 处理城市 {city_name} 时出错: {e}")
            traceback.print_exc()


# ==========================================
# 道路曲率 (sinuosity) 计算
# ==========================================
def compute_geometry_euclidean(gdf):
    def get_coords(geom):
        if geom is None or geom.is_empty:
            return np.nan, np.nan, np.nan, np.nan

        # 处理 LineString
        if geom.geom_type == 'LineString':
            coords = geom.coords
            if len(coords) < 2:
                return np.nan, np.nan, np.nan, np.nan
            return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]

        # 处理 MultiLineString (Overlay 可能会产生这种情况)
        elif geom.geom_type == 'MultiLineString':
            if len(geom.geoms) == 0:
                return np.nan, np.nan, np.nan, np.nan
            # 简单起见，取最长的一段来代表方向和起止
            longest = max(geom.geoms, key=lambda g: g.length)
            coords = longest.coords
            if len(coords) < 2:
                return np.nan, np.nan, np.nan, np.nan
            return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]
        else:
            return np.nan, np.nan, np.nan, np.nan

    # 提取坐标
    coords = gdf.geometry.apply(get_coords).tolist()
    coords_df = pd.DataFrame(coords, columns=['x1', 'y1', 'x2', 'y2'], index=gdf.index)

    # 计算欧氏距离 sqrt(dx^2 + dy^2)
    dist = np.sqrt(
        (coords_df['x2'] - coords_df['x1']) ** 2 +
        (coords_df['y2'] - coords_df['y1']) ** 2
    )
    return dist


def process_single_road_sinuosity(city_name, grid_file, osm_dir, indicator_name="road_sinuosity"):
    output_csv = PATHS["RESULTS"] / indicator_name / f"{city_name}.csv"
    if output_csv.exists():
        print(f"  [跳过] {city_name}.csv 已存在")
        return

    # 1. 读取网格
    grid_gdf = gpd.read_file(grid_file)
    if grid_gdf.empty:
        return

    # 2. 加载路网
    try:
        edges_gdf = merge_city_road_graph(osm_dir, city_name)
    except Exception as e:
        print(f"  ❌ 加载路网失败 {city_name}: {e}")
        return

    if edges_gdf.empty:
        print(f"  ⚠️ {city_name} 路网为空，生成默认值。")
        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['sinuosity'] = 1.0  # 无路网视为直线(或无弯曲)
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        return

    # 3. 统一投影到 UTM (计算长度必须在投影坐标系下)
    utm_crs = grid_gdf.estimate_utm_crs()
    grid_utm = grid_gdf.to_crs(utm_crs)
    edges_utm = edges_gdf.to_crs(utm_crs)

    # 4. 空间裁剪 (Overlay)
    print(f"  - 正在计算 {city_name} 的路网曲率 (Overlay)...")

    grid_subset = grid_utm[['grid_id', 'geometry']]
    edges_subset = edges_utm[['geometry']]

    try:
        # Intersection: 将路网按网格切断
        intersections = gpd.overlay(
            edges_subset,
            grid_subset,
            how='intersection',
            keep_geom_type=False
        )
    except Exception as e:
        print(f"  ❌ Overlay 计算失败: {e}")
        return

    if intersections.empty:
        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['sinuosity'] = 1.0
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        return

    # 关键：Overlay 可能会产生 MultiLineString，炸开成简单的 LineString 以便准确计算首尾坐标
    intersections = intersections.explode(index_parts=False)

    # 5. 计算属性
    # A. 实际路径长度 (沿路网)
    intersections['actual_len'] = intersections.geometry.length

    # B. 直线距离 (首尾欧氏距离)
    intersections['euclidean_len'] = compute_geometry_euclidean(intersections)

    # 6. 分组聚合计算 (Vectorized GroupBy)
    print(f"  - 正在聚合计算 Sinuosity...")

    # 按 grid_id 汇总长度
    # Sinuosity = Sum(Actual Length) / Sum(Euclidean Length)
    # 这种聚合方式比 "先算每条路的sinuosity再平均" 更稳健，能避免短路段的极端值影响
    stats = intersections.groupby('grid_id')[['actual_len', 'euclidean_len']].sum().reset_index()

    # 计算比率
    # 避免分母为 0 (如果全是闭合环路，欧氏距离为0)
    # 如果 euclidean_len 极小，说明主要是环路，给予一个合理的上限或保持原值
    stats['sinuosity'] = np.where(
        stats['euclidean_len'] > 1e-3,
        stats['actual_len'] / stats['euclidean_len'],
        1.0  # 如果分母为0（如纯圆环），或者没有路，暂定为1或可设为 NaN
    )

    # 理论上 Sinuosity 最小为 1。由于浮点数误差，可能出现 0.999
    stats.loc[stats['sinuosity'] < 1.0, 'sinuosity'] = 1.0

    # 7. 合并回原始网格
    final_df = grid_utm.merge(stats[['grid_id', 'sinuosity']], on='grid_id', how='left')

    # 填充 NaN (没有路网的格子，曲率设为 1，即最平直/无复杂性)
    final_df['sinuosity'] = final_df['sinuosity'].fillna(1.0)
    final_df['sinuosity'] = final_df['sinuosity'].round(4)

    # 8. 保存
    output_df = final_df[['grid_id', 'is_deployed', 'sinuosity']]
    os.makedirs(output_csv.parent, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def compute_sinuosity(indicator_name="sinuosity"):
    print("\n>>> 开始按城市计算道路曲率 <<<")

    # 确保输出目录存在
    os.makedirs(PATHS["RESULTS"] / indicator_name, exist_ok=True)

    # 获取所有城市名
    city_dirs = [d for d in os.listdir(PATHS["GEOJSON"]) if d.endswith(".geojson")]
    print(f"📍 共识别到 {len(city_dirs)} 个城市")

    for city_dir in sorted(city_dirs):
        city_name = city_dir.split(".")[0]
        # 注意：这里假设你有 grid 的 geojson 文件
        grid_file = PATHS['GRID_OUTPUT'] / f"{city_name}.geojson"

        if not grid_file.exists():
            print(f"⚠️  跳过 {city_name}：缺少网格文件")
            continue

        try:
            process_single_road_sinuosity(
                city_name=city_name,
                grid_file=grid_file,
                osm_dir=PATHS["ENTROPY"],  # 假设路网文件在这个目录下
                indicator_name=indicator_name
            )
        except Exception as e:
            print(f"❌ 处理城市 {city_name} 时出错: {e}")
            traceback.print_exc()


# ==========================================
# 复杂交叉口比例计算
# ==========================================
def get_valid_nodes_and_roundabouts(G, road_types=ROAD_TYPES, min_street_count=3):
    """
    提取有效普通交叉口节点和环岛组件。
    与 calculate_inter_number 函数保持一致的识别逻辑。
    """
    # 确保 road_types 是集合，避免数组布尔运算错误
    target_road_types = set(road_types)

    # ===== 1. 识别环岛（基于边的 junction 属性） =====
    roundabout_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get('junction') == 'roundabout':
            roundabout_edges.append((u, v, key))

    roundabout_nodes = set()
    roundabout_components = []

    if roundabout_edges:
        try:
            # 创建环岛子图获取所有环岛节点
            roundabout_subgraph = G.edge_subgraph(roundabout_edges)
            roundabout_nodes = set(roundabout_subgraph.nodes())

            # 构建环岛组件（将相连的环岛节点视为一个实体）
            roundabout_subgraph_undirected = roundabout_subgraph.to_undirected()
            roundabout_components = list(nx.connected_components(roundabout_subgraph_undirected))
        except Exception as e:
            # 如果edge_subgraph失败，使用备选方案
            print(f"Warning: edge_subgraph failed, using alternative method: {e}")
            # 备选方案：收集所有环岛节点并手动计算连通分量
            for u, v, key in roundabout_edges:
                roundabout_nodes.add(u)
                roundabout_nodes.add(v)

            if roundabout_nodes:
                roundabout_subgraph = G.subgraph(roundabout_nodes).to_undirected()
                roundabout_components = list(nx.connected_components(roundabout_subgraph))
    else:
        roundabout_components = []

    # ===== 2. 预处理边属性 =====
    # 注意：直接遍历 G.edges 比转 GeoDataFrame 更快且不易出错
    node_road_types = defaultdict(set)

    for u, v, data in G.edges(data=True):
        hw = data.get('highway', '')
        # 处理 highway 可能是 list 或 string 的情况
        if isinstance(hw, list):
            for h in hw:
                if h:  # 跳过空值
                    node_road_types[u].add(h)
                    node_road_types[v].add(h)
        elif isinstance(hw, str) and hw:
            node_road_types[u].add(hw)
            node_road_types[v].add(hw)

    # ===== 3. 筛选普通交叉口 =====
    valid_nodes = set()
    for n in G.nodes():
        # 排除环岛上的点
        if n in roundabout_nodes:
            continue

        # 获取度数 (street_count)
        # G.degree(n) 在 MultiGraph 中计算的是边的数量，可能比物理路口数多
        # 使用 neighbors 的数量更准确地代表"路口的分支数"
        degree = len(list(G.neighbors(n)))

        if degree < min_street_count:
            continue

        # 检查道路类型匹配
        # 只要该节点连接的道路类型中，有任意一种属于 target_road_types 即可
        connected_types = node_road_types[n]

        # 检查是否有交集
        if not connected_types.isdisjoint(target_road_types):
            valid_nodes.add(n)

    return valid_nodes, roundabout_components


def process_single_ratio_complex(city_name, grid_file, osm_dir, indicator_name="ratio_complex"):
    output_csv = PATHS["RESULTS"] / indicator_name / f"{city_name}.csv"
    if output_csv.exists():
        print(f"  [跳过] {city_name}.csv 已存在")
        return

    # 1. 读取网格
    try:
        grid_gdf = gpd.read_file(grid_file)
    except Exception:
        print(f"  ❌ 无法读取网格文件 {grid_file}")
        return

    if grid_gdf.empty:
        return

    # 2. 查找 graphml 文件
    target_file = None
    found = False
    for root, dirs, files in os.walk(osm_dir):
        for f in files:
            if f.endswith(".graphml") and city_name in f:
                target_file = os.path.join(root, f)
                found = True
                break
        if found:
            break

    if not target_file:
        print(f"  ⚠️ 未找到 {city_name} 的 .graphml 文件")
        return

    try:
        # 加载图
        G = ox.load_graphml(target_file)
        # 转无向图，方便计算邻居数
        G = ox.convert.to_undirected(G)
    except Exception as e:
        print(f"  ❌ 加载路网失败 {city_name}: {e}")
        return

    # 3. 获取有效节点和环岛
    try:
        valid_nodes, roundabout_components = get_valid_nodes_and_roundabouts(G)
    except Exception as e:
        print(f"  ❌ 分析节点逻辑出错 {city_name}: {e}")
        traceback.print_exc()
        return

    # 4. 构建 GeoDataFrame 数据源
    node_points = []

    # 4a. 普通交叉口
    for n in valid_nodes:
        # 确保节点有坐标
        if 'x' not in G.nodes[n] or 'y' not in G.nodes[n]:
            continue

        x, y = G.nodes[n]['x'], G.nodes[n]['y']

        # 计算是否为多岔路口 (Multi-arm: > 4)
        degree = len(list(G.neighbors(n)))
        is_multi_arm = degree > 4

        node_points.append({
            'geometry': Point(x, y),
            'is_complex': is_multi_arm,
            'type': 'normal'
        })

    # 4b. 环岛 (视为复杂路口)
    for comp in roundabout_components:
        coords = []
        for n in comp:
            if 'x' in G.nodes[n] and 'y' in G.nodes[n]:
                coords.append((G.nodes[n]['x'], G.nodes[n]['y']))

        if not coords:
            continue

        if len(coords) == 1:
            pt = Point(coords[0])
        else:
            # 计算这一组点的中心
            multipoint = gpd.points_from_xy([c[0] for c in coords], [c[1] for c in coords])
            pt = gpd.GeoSeries(multipoint).unary_union.centroid

        node_points.append({
            'geometry': pt,
            'is_complex': True,  # 环岛本身算作复杂结构
            'type': 'roundabout'
        })

    # 如果没有找到任何交叉口
    if not node_points:
        print(f"  ⚠️ {city_name} 无有效交叉口，生成零值结果。")
        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['ratio_complex'] = 0.0
        final_df['intersection_density'] = 0.0
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        return

    # 构建 GeoDataFrame
    inters_gdf = gpd.GeoDataFrame(node_points, crs="EPSG:4326")

    # 5. 投影转换 (用于 Sjoin)
    try:
        utm_crs = grid_gdf.estimate_utm_crs()
        grid_utm = grid_gdf.to_crs(utm_crs)
        inters_utm = inters_gdf.to_crs(utm_crs)
    except Exception as e:
        print(f"  ❌ 投影转换失败 {city_name}: {e}")
        return

    # 6. 空间连接
    # print(f"  - 计算 {city_name} (Sjoin)...")
    joined = gpd.sjoin(inters_utm, grid_utm[['grid_id', 'geometry']], how='inner', predicate='within')

    # 7. 统计聚合
    stats = joined.groupby('grid_id').agg(
        total_intersections=('is_complex', 'count'),
        complex_intersections=('is_complex', 'sum')
    ).reset_index()

    # 计算比例
    stats['ratio_complex'] = 0.0
    mask = stats['total_intersections'] > 0
    stats.loc[mask, 'ratio_complex'] = (
            stats.loc[mask, 'complex_intersections'] / stats.loc[mask, 'total_intersections'] * 100
    )

    # 8. 合并回原始网格
    final_df = grid_utm.merge(stats, on='grid_id', how='left')

    # 填充空值
    final_df['ratio_complex'] = final_df['ratio_complex'].fillna(0.0).round(2)
    final_df['total_intersections'] = final_df['total_intersections'].fillna(0)
    final_df['complex_intersections'] = final_df['complex_intersections'].fillna(0)

    # 9. 计算密度 (个/km2)
    area_km2 = final_df.geometry.area / 1e6
    final_df['intersection_density'] = (
            final_df['total_intersections'] / area_km2
    ).replace([np.inf, -np.inf], 0).fillna(0).round(2)

    # 10. 保存结果
    output_df = final_df[['grid_id', 'is_deployed', 'ratio_complex', 'intersection_density']]
    os.makedirs(output_csv.parent, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def compute_ratio_complex(indicator_name="ratio_complex"):
    print("\n>>> 开始按城市计算路口复杂性 <<<")
    os.makedirs(PATHS["RESULTS"] / indicator_name, exist_ok=True)

    city_files = [f for f in os.listdir(PATHS["GEOJSON"]) if f.endswith(".geojson")]
    print(f"📍 共识别到 {len(city_files)} 个城市")

    for city_file in sorted(city_files):
        city_name = city_file.split(".")[0]
        grid_file = PATHS['GRID_OUTPUT'] / f"{city_name}.geojson"
        if not grid_file.exists():
            print(f"⚠️  跳过 {city_name}：缺少网格文件")
            continue

        try:
            process_single_ratio_complex(
                city_name=city_name,
                grid_file=grid_file,
                osm_dir=PATHS["ENTROPY"],
                indicator_name=indicator_name
            )
        except Exception as e:
            print(f"❌ 处理城市 {city_name} 时出错: {e}")
            traceback.print_exc()


# ==========================================
# 距离最近公交站点的距离计算
# ==========================================
def get_or_download_stations(city_name, search_gdf, cache_dir):
    station_file = cache_dir / f"{city_name}.geojson"

    # --- A. If file exists, read it ---
    if station_file.exists():
        try:
            gdf = gpd.read_file(station_file)
            print(f"  ✅ Loaded from cache: {city_name}")
            return gdf
        except Exception as e:
            print(f"  ⚠️ Local file corrupted, re-downloading: {e}")
            station_file.unlink(missing_ok=True)

    # --- B. If file does not exist, download it ---
    print(f"  ⬇️ Downloading transport stations for {city_name} from OSM...")

    # Configure OSMnx settings for better stability
    ox.settings.use_cache = True
    ox.settings.log_console = False
    # Increase timeout (default is often too short for large cities like Shanghai)
    ox.settings.requests_timeout = 180

    # List of Overpass API mirrors to try
    overpass_endpoints = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter"
    ]

    # 1. Ensure search area is WGS84
    search_polygon = search_gdf.to_crs(epsg=4326).unary_union.convex_hull

    # Retry loop over different endpoints
    for endpoint in overpass_endpoints:
        ox.settings.overpass_endpoint = endpoint
        print(f"  🔄 Trying endpoint: {endpoint}")

        try:
            # 2. Download using osmnx
            stations = ox.features_from_polygon(search_polygon, tags=POI_STATION_TAGS)

            if stations.empty:
                print(f"  ⚠️ OSM returned empty data for: {city_name}")
                return gpd.GeoDataFrame()

            # 3. Data Cleaning: Keep geometry, convert to points
            stations = stations[['geometry']].copy()
            # Use centroid to convert Polygons (like large stations) to Points
            stations['geometry'] = stations.geometry.centroid

            # 4. Ensure CRS is WGS84
            if stations.crs is None:
                stations.set_crs(epsg=4326, inplace=True)
            else:
                stations = stations.to_crs(epsg=4326)

            # 5. Save to local cache
            os.makedirs(cache_dir, exist_ok=True)
            stations.to_file(station_file, driver='GeoJSON')

            print(f"  ✅ Successfully saved: {city_name}")
            return stations

        except (ConnectionError, ReadTimeout, SSLError) as e:
            print(f"  ⚠️ Network error with {endpoint}: {e}")
            print("  ⏳ Waiting 5 seconds before trying next mirror...")
            time.sleep(5)
            continue  # Try the next endpoint in the list
        except Exception as e:
            # If it's a non-network error (e.g., logic error), stop trying
            print(f"  ❌ Critical error processing {city_name}: {e}")
            return gpd.GeoDataFrame()

    print(f"  ❌ All mirrors failed for {city_name}.")
    return gpd.GeoDataFrame()


def calculate_nearest_station_distance(grid_gdf, stations_gdf):
    EARTH_RADIUS_KM = 6371.0

    # 1. 准备网格数据 (确保转换为 WGS84 以获取经纬度)
    grid_wgs84 = grid_gdf.to_crs(epsg=4326)
    grid_centroids = grid_wgs84.geometry.centroid

    # BallTree 需要 [lat, lon] 的弧度数组
    grid_coords = np.radians(np.column_stack([grid_centroids.y, grid_centroids.x]))

    # 2. 准备站点数据 (已经是 WGS84 Point)
    station_coords = np.radians(np.column_stack([stations_gdf.geometry.y, stations_gdf.geometry.x]))

    # 3. 构建 BallTree
    # metric='haversine' 专门用于球面距离
    tree = BallTree(station_coords, leaf_size=40, metric='haversine')

    # 4. 查询最近邻 (k=1)
    dist_radians, _ = tree.query(grid_coords, k=1)

    # 5. 转为千米
    dist_km = dist_radians.flatten() * EARTH_RADIUS_KM

    return dist_km


def process_single_station_distance(city_name, grid_file, osm_cache_dir, indicator_name="dist_to_station"):
    output_csv = PATHS["RESULTS"] / indicator_name / f"{city_name}.csv"

    # 结果已存在则跳过
    if output_csv.exists():
        print(f"  [跳过] {city_name}.csv 已存在")
        return

    # 1. 读取网格
    grid_gdf = gpd.read_file(grid_file)
    if grid_gdf.empty:
        return

    # 2. 获取或下载站点数据
    # 传入 grid_gdf 用于界定下载范围
    stations_gdf = get_or_download_stations(city_name, grid_gdf, osm_cache_dir)

    # 3. 处理无站点情况
    if stations_gdf.empty:
        print(f"  ⚠️ {city_name} 无轨道交通站点数据，标记为 NaN（无服务）")
        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['dist_to_station'] = np.nan
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        return

    # 4. 计算距离
    try:
        dists = calculate_nearest_station_distance(grid_gdf, stations_gdf)

        final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
        final_df['dist_to_station'] = dists
        final_df['dist_to_station'] = final_df['dist_to_station'].round(4)

        # 保存
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"  ❌ 计算出错 {city_name}: {e}")
        traceback.print_exc()


def compute_station_distance(indicator_name="dist_to_station"):
    print("\n>>> 开始计算：网格到最近轨道交通站点距离 <<<")

    os.makedirs(PATHS["RESULTS"] / indicator_name, exist_ok=True)

    city_dirs = [d for d in os.listdir(PATHS["GEOJSON"]) if d.endswith(".geojson")]
    print(f"📍 待处理城市: {len(city_dirs)} 个")

    for city_dir in sorted(city_dirs):
        city_name = city_dir.split(".")[0]
        grid_file = PATHS['GRID_OUTPUT'] / f"{city_name}.geojson"

        if not grid_file.exists():
            continue

        try:
            process_single_station_distance(
                city_name=city_name,
                grid_file=grid_file,
                osm_cache_dir=PATHS["OSM_STATION"],
                indicator_name=indicator_name
            )
        except Exception as e:
            print(f"❌ 处理城市 {city_name} 致命错误: {e}")


# ==========================================
# POI丰度计算
# ==========================================
def download_single_city_poi(city_name, grid_file):
    cache_path = PATHS["POI_DIR"] / f"{city_name}.parquet"

    if cache_path.exists():
        return f"✅ [跳过下载] {city_name} 缓存已存在"

    try:
        # 1. 读取网格
        grid_gdf = gpd.read_file(grid_file)
        if grid_gdf.empty:
            return f"⚠️ [跳过下载] {city_name} 网格为空"

        minx, miny, maxx, maxy = grid_gdf.total_bounds
        search_area = box(minx, miny, maxx, maxy)

        # 2. 处理 Tag
        current_tags = POI_TAGS.copy()
        if city_name == "Austin":
            current_tags.pop('leisure', None)

        # 3. 下载 POI
        pois = ox.features_from_polygon(search_area, tags=current_tags)

        # 确保目录存在
        os.makedirs(cache_path.parent, exist_ok=True)

        if pois.empty:
            empty_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')
            empty_gdf.to_parquet(cache_path)
            return f"⚠️ [下载为空] {city_name} 未找到 POI"

        # 1. 重置索引
        pois = pois.reset_index()

        # 2. 筛选列
        desired_columns = ['geometry', 'element_type', 'osmid'] + list(current_tags.keys())
        cols_to_keep = [c for c in desired_columns if c in pois.columns]
        pois = pois[cols_to_keep]

        # 3. 强制类型转换
        for col in pois.columns:
            if col != 'geometry':
                pois[col] = pois[col].astype(str)

        # 4. 保存
        pois.to_parquet(cache_path)
        return f"⬇️ [下载完成] {city_name}: {len(pois)} 条"

    except Exception as e:
        return f"❌ [下载失败] {city_name}: {str(e)[:150]}"


def process_single_poi_richness_from_cache(city_name, grid_file, geojson_path):
    try:
        # 定义两个输出路径
        output_csv = PATHS["RESULTS"] / "poi_richness" / f"{city_name}.csv"
        output_csv_log = PATHS["RESULTS"] / "poi_richness_log" / f"{city_name}.csv"

        cache_path = PATHS["POI_DIR"] / f"{city_name}.parquet"

        if output_csv.exists() and output_csv_log.exists():
            return f"✅ [跳过计算] {city_name} 结果已存在"

        # 1. 读取网格
        try:
            grid_gdf = gpd.read_file(grid_file)
        except Exception as e:
            return f"❌ [失败] {city_name} 读取网格失败: {e}"

        if grid_gdf.empty:
            return f"❌ [失败] {city_name} 网格为空"

        # 2. 读取本地缓存的 POI
        if not cache_path.exists():
            return f"❌ [失败] {city_name} 缺少缓存文件，请先运行下载阶段"

        try:
            pois = gpd.read_parquet(cache_path)
        except Exception as e:
            return f"❌ [失败] {city_name} 读取缓存 Parquet 失败: {e}"

        if pois.empty:
            _save_empty_result(grid_gdf, output_csv, output_csv_log)
            return f"⚠️ [警告] {city_name} 缓存数据为空"

        # 3. 预处理 (投影转换)
        try:
            target_crs = grid_gdf.estimate_utm_crs()
            grid_utm = grid_gdf.to_crs(target_crs)[['grid_id', 'geometry']]
            pois_utm = pois.to_crs(target_crs)
            pois_utm['geometry'] = pois_utm.geometry.centroid
        except Exception as e:
            return f"❌ [错误] {city_name} 投影/质心转换失败: {e}"

        # 4. 空间连接
        try:
            joined_raw = gpd.sjoin(pois_utm, grid_utm, how='inner', predicate='within')
        except Exception as e:
            return f"❌ [错误] {city_name} 空间连接失败: {e}"

        if joined_raw.empty:
            _save_empty_result(grid_gdf, output_csv, output_csv_log)
            return f"⚠️ [警告] {city_name} 空间匹配后无数据"

        # 5. 标签展开
        current_tags = POI_TAGS.copy()
        if city_name == "Austin":
            current_tags.pop('leisure', None)

        available_tag_cols = [col for col in current_tags.keys() if col in joined_raw.columns]

        if not available_tag_cols:
            _save_empty_result(grid_gdf, output_csv, output_csv_log)
            return f"⚠️ [警告] {city_name} 无指定标签列"

        subset = joined_raw[['grid_id'] + available_tag_cols].copy()
        subset['tmp_id'] = range(len(subset))

        stacked = subset.set_index(['grid_id', 'tmp_id']).stack().reset_index()
        stacked.columns = ['grid_id', 'tmp_id', 'tag_key', 'tag_value']

        # 清理空值
        valid_pois = stacked[~stacked['tag_value'].isin(['nan', 'None', '', 'NULL'])]

        # --- 保存 POI 明细 ---
        poi_detail_csv = PATHS["POI_DIR"] / f"{city_name}.csv"
        os.makedirs(poi_detail_csv.parent, exist_ok=True)

        if not valid_pois.empty:
            detail_df = valid_pois[['grid_id', 'tag_value']].copy()
            detail_df.rename(columns={'tag_value': 'category'}, inplace=True)
            detail_df.to_csv(poi_detail_csv, index=False, encoding='utf-8-sig')
        else:
            pd.DataFrame(columns=['grid_id', 'category']).to_csv(poi_detail_csv, index=False)

        # 6. 统计
        stats = valid_pois.groupby('grid_id').agg(
            poi_count=('tag_value', 'count'),
            poi_richness=('tag_value', 'nunique')
        ).reset_index()

        # 7. 合并与保存 (原始数据)
        final_df = grid_gdf[['grid_id', 'is_deployed']].merge(stats, on='grid_id', how='left')
        final_df['poi_count'] = final_df['poi_count'].fillna(0).astype(int)
        final_df['poi_richness'] = final_df['poi_richness'].fillna(0).astype(int)

        # 保存原始版本 (列名保持 poi_richness)
        os.makedirs(output_csv.parent, exist_ok=True)
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        # 8. 计算对数并保存 (Log 版本)
        final_df_log = final_df.copy()
        final_df_log['poi_richness'] = np.log1p(final_df_log['poi_richness'])
        final_df_log.rename(columns={'poi_richness': 'poi_richness_log'}, inplace=True)

        os.makedirs(output_csv_log.parent, exist_ok=True)
        final_df_log.to_csv(output_csv_log, index=False, encoding='utf-8-sig')

        max_count = final_df['poi_count'].max()
        coverage = (final_df['poi_count'] > 0).mean()
        return f"✅ [计算完成] {city_name}: Max={max_count}, Cov={coverage:.1%}"

    except Exception as e:
        return f"❌ [严重崩溃] {city_name}: {str(e)}"


def _save_empty_result(grid_gdf, output_path, output_path_log):
    final_df = grid_gdf[['grid_id', 'is_deployed']].copy()
    final_df['poi_count'] = 0
    final_df['poi_richness'] = 0

    # 保存原始空文件
    os.makedirs(output_path.parent, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    # 保存对数空文件
    final_df_log = final_df.copy()
    final_df_log.rename(columns={'poi_richness': 'poi_richness_log'}, inplace=True)

    os.makedirs(output_path_log.parent, exist_ok=True)
    final_df_log.to_csv(output_path_log, index=False)


def compute_poi_richness(indicator_name="poi_richness"):
    os.makedirs(PATHS["RESULTS"] / indicator_name, exist_ok=True)
    os.makedirs(PATHS["RESULTS"] / f"{indicator_name}_log", exist_ok=True)

    # 1. 准备任务列表
    city_files = [f for f in os.listdir(PATHS["GEOJSON"]) if f.endswith(".geojson")]
    tasks = []
    for f in city_files:
        city_name = f.split(".")[0]
        grid_file = PATHS['GRID_OUTPUT'] / f"{city_name}.geojson"
        geojson_path = PATHS["GEOJSON"] / f
        if grid_file.exists():
            tasks.append((city_name, grid_file, geojson_path))

    print(f"📍 共识别到 {len(tasks)} 个城市任务")

    # ---------------------------------------------------------
    # 阶段 1: 串行下载
    # ---------------------------------------------------------
    print("\n [阶段 1/2] 开始下载 POI 数据")

    with tqdm(total=len(tasks), desc="Downloading POIs") as pbar:
        for city_name, grid_file, _ in tasks:
            res = download_single_city_poi(city_name, grid_file)
            if "❌" in res:
                print(f"\n{res}")
            pbar.set_postfix_str(res.split("]")[-1].strip()[:20])
            pbar.update(1)

    # ---------------------------------------------------------
    # 阶段 2: 并行计算
    # ---------------------------------------------------------
    print("\n [阶段 2/2] 开始本地计算")

    max_workers = min(os.cpu_count(), 8)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_city = {executor.submit(process_single_poi_richness_from_cache, *t): t[0] for t in tasks}

        with tqdm(total=len(tasks), desc="Processing Data") as pbar:
            for future in concurrent.futures.as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    res = future.result()
                    msg = res.split("]")[-1].strip() if "]" in res else res
                    pbar.set_postfix_str(msg)
                    results.append(res)
                except Exception as exc:
                    error_msg = f"❌ 城市 {city} 进程异常: {exc}"
                    print(f"\n{error_msg}")
                    results.append(error_msg)
                pbar.update(1)

    print("\n--- 最终汇总 ---")
    error_count = 0
    for res in results:
        if "❌" in res:
            print(res)
            error_count += 1

    if error_count == 0:
        print("🎉 所有任务顺利完成！")
    else:
        print(f"⚠️ 完成，但有 {error_count} 个错误，请检查日志。")


# ==========================================
# 绘图
# ==========================================
# Hyperparameters of scienceplots
plt.style.use(['science', 'no-latex', 'nature'])

plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'legend.title_fontsize': 24,
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

def draw_scale_bar(ax, bounds, crs):
    minx, maxx, miny, maxy = bounds

    # 1. 计算当前视图的中心纬度（用于经纬度转米）
    center_y = (miny + maxy) / 2

    # 2. 估算当前视图宽度的实际米数
    if crs.is_geographic:
        # 简易估算：1度经度在特定纬度下的米数
        # 公式：111319.55 * cos(lat_rad)
        meters_per_unit = 111319.55 * math.cos(math.radians(center_y))
    else:
        # 假设投影坐标系单位本身就是米
        meters_per_unit = 1

    view_width_m = (maxx - minx) * meters_per_unit

    # 3. 决定比例尺代表的距离 (目标是占总宽度的 20% 左右)
    target_scale_m = view_width_m * 0.2

    # 候选刻度 (米)
    candidates = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    # 找到最接近的候选值
    bar_meters = min(candidates, key=lambda x: abs(x - target_scale_m))

    # 4. 将比例尺长度转换回地图单位 (度 或 米)
    bar_width_map_units = bar_meters / meters_per_unit

    # 5. 确定绘制位置 (左下角，留出 padding)
    # padding 设为视窗宽高的 5%
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05

    start_x = minx + pad_x
    start_y = miny + pad_y
    end_x = start_x + bar_width_map_units

    # 6. 绘制图形
    # 刻度线高度 (视窗高度的 1.5%)
    tick_height = (maxy - miny) * 0.015

    # 绘制线段 (形状: |____|)
    # 使用 plot 绘制线条: [x序列], [y序列]
    # 顺序: 左上 -> 左下 -> 右下 -> 右上
    x_coords = [start_x, start_x, end_x, end_x]
    y_coords = [start_y + tick_height, start_y, start_y, start_y + tick_height]

    ax.plot(x_coords, y_coords, color='white', linewidth=2, zorder=20)

    # 添加阴影/描边效果增强对比度
    ax.plot(x_coords, y_coords, color='black', linewidth=4, zorder=19, alpha=0.3)

    # 7. 绘制文字
    if bar_meters >= 1000:
        label_text = f"{int(bar_meters / 1000)} km"
    else:
        label_text = f"{int(bar_meters)} m"

    text_x = (start_x + end_x) / 2
    # 文字位置在线条上方一点
    text_y = start_y + tick_height * 1.5

    t = ax.text(
        text_x, text_y,
        label_text,
        ha='center', va='bottom',
        color='white',
        fontsize=24,
        fontweight='bold',
        zorder=20
    )
    # 给文字加黑色描边
    t.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])


def get_square_bounds(gdf, aspect_ratio=1.0, padding_factor=0.1):
    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny

    # 计算中心点
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    # 1. 确定包含城市所需的最小边框（带 padding）
    max_side = max(width, height)
    base_side = max_side * (1 + padding_factor)

    # 2. 根据目标长宽比计算最终的地理宽高
    if aspect_ratio >= 1:
        final_height = base_side
        final_width = base_side * aspect_ratio
    else:
        final_width = base_side
        final_height = base_side / aspect_ratio

    # 3. 重新计算边界
    new_minx = cx - final_width / 2
    new_maxx = cx + final_width / 2
    new_miny = cy - final_height / 2
    new_maxy = cy + final_height / 2

    return new_minx, new_maxx, new_miny, new_maxy


def generate_overlay_maps(indicator, geojson_folder, raster_path, output_folder):
    STYLE = {
        'bg_color': '#1a1a1a',
        'grid_color': '#ffffff',
        'grid_alpha': 0.15,
        'grid_width': 0.5,
        'deploy_fill': "#c8c2c2",   # 中性灰
        'deploy_edge': "#c8c2c2",   # 边缘用稍微亮一点的灰色
        'deploy_alpha': 0.15,       # 不透明度
        'text_color': 'white',
        'cmap': 'magma'
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    city_files = glob.glob(os.path.join(geojson_folder, "*.geojson"))

    FIG_WIDTH = 9  # 英寸
    FIG_HEIGHT = 9  # 英寸
    DPI = 300  # 分辨率

    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        src_nodata = src.nodata

        for i, city_file in enumerate(city_files):
            try:
                city_name = os.path.splitext(os.path.basename(city_file))[0]
                print(f"[{i + 1}/{len(city_files)}] 处理 {city_name}")

                # --- 1. 数据读取 ---
                gdf = gpd.read_file(city_file)
                if gdf.crs != src_crs:
                    gdf = gdf.to_crs(src_crs)
                if len(gdf) == 0: continue

                if len(gdf) > 1:
                    geometry = [gdf.geometry.unary_union]
                else:
                    geometry = [gdf.geometry.iloc[0]]

                # --- 2. 裁剪栅格 ---
                try:
                    out_image, out_transform = mask(src, geometry, crop=True, all_touched=True)
                    raster_data = out_image[0]
                except Exception as e:
                    print(f"  裁剪失败: {e}")
                    continue

                if src_nodata is not None:
                    raster_data = np.where(raster_data == src_nodata, np.nan, raster_data)
                raster_data = np.where(raster_data <= 0, np.nan, raster_data)

                valid_mask = ~np.isnan(raster_data)
                if not np.any(valid_mask):
                    print(f"  无有效数据，跳过")
                    continue

                valid_pixels = raster_data[valid_mask]

                # --- 3. 准备部署数据 ---
                deployed_gdf = None
                if 'is_deployed' in gdf.columns:
                    try:
                        deployed_gdf = gdf[gdf['is_deployed'].astype(int) == 1].copy()
                    except:
                        pass

                # --- 4. 绘图初始化 ---
                fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
                ax = fig.add_axes([0, 0, 1, 1], facecolor=STYLE['bg_color'])
                ax.set_axis_off()

                # --- 5. 绘制栅格 ---
                vmin = np.nanpercentile(valid_pixels, 2)
                vmax = np.nanpercentile(valid_pixels, 99)
                norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

                extent = plotting_extent(raster_data, out_transform)

                im = ax.imshow(
                    raster_data,
                    extent=extent,
                    cmap=STYLE['cmap'],
                    norm=norm,
                    alpha=1.0,
                    interpolation='bilinear',
                    zorder=1
                )

                # --- 6. 绘制矢量 ---
                gdf.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor=STYLE['grid_color'],
                    linewidth=STYLE['grid_width'],
                    alpha=STYLE['grid_alpha'],
                    zorder=2
                )

                # gdf.dissolve().plot(
                #     ax=ax,
                #     facecolor='none',
                #     edgecolor=STYLE['grid_color'],
                #     linewidth=1.0,
                #     alpha=0.6,
                #     zorder=2.5
                # )

                # --- 7. 绘制部署区域 ---
                if deployed_gdf is not None and not deployed_gdf.empty:
                    deployed_gdf.plot(
                        ax=ax,
                        facecolor=STYLE['deploy_fill'],
                        edgecolor='none',
                        alpha=STYLE['deploy_alpha'],
                        zorder=3
                    )
                    deployed_gdf.plot(
                        ax=ax,
                        facecolor='none',
                        edgecolor=STYLE['deploy_edge'],
                        linewidth=1.2,
                        zorder=4
                    )

                # --- 8. 调整视野 ---
                target_minx, target_maxx, target_miny, target_maxy = get_square_bounds(
                    gdf, aspect_ratio=FIG_WIDTH / FIG_HEIGHT, padding_factor=0.15
                )
                ax.set_xlim(target_minx, target_maxx)
                ax.set_ylim(target_miny, target_maxy)

                # --- 9. 绘制比例尺 ---
                draw_scale_bar(
                    ax,
                    (target_minx, target_maxx, target_miny, target_maxy),
                    src_crs
                )

                # --- 10. Colorbar ---
                cax = inset_axes(
                    ax,
                    width="5%",
                    height="40%",
                    loc='center right',
                    bbox_to_anchor=(0, 0, 0.9, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0
                )
                cbar = plt.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.yaxis.set_tick_params(color=STYLE['text_color'], labelcolor=STYLE['text_color'], labelsize=24)
                cbar.outline.set_visible(False)

                label_text = LABELS.get(indicator, indicator)
                cbar.set_label(label_text, color=STYLE['text_color'], size=26, labelpad=10)

                # --- 11. 标题与图例 ---
                title_x_offset = 0  # 默认值
                legend_x_offset = 1.05  # 默认值

                if city_name == "Hamburg":
                    title_x_offset = -0.3  # 向右移动标题
                    legend_x_offset = 1.30  # 向右移动图例，避免与色带重叠
                elif city_name == "San Francisco":
                    title_x_offset = -0.1  # 向右移动标题
                    legend_x_offset = 1.1  # 向右移动图例，避免与色带重叠

                title_text = ax.text(
                    title_x_offset, 0.9, city_name,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=28,
                    fontweight='bold',
                    color=STYLE['text_color'],
                    zorder=10
                )
                title_text.set_path_effects([pe.withStroke(linewidth=3, foreground=STYLE['bg_color'])])

                if deployed_gdf is not None and not deployed_gdf.empty:
                    legend_elements = [
                        Patch(
                            facecolor=mcolors.to_rgba(STYLE['deploy_fill'], STYLE['deploy_alpha']),
                            edgecolor=STYLE['deploy_edge'],
                            linewidth=1.2,
                            label='AV-served area'
                        )
                    ]
                    leg = ax.legend(
                        handles=legend_elements,
                        loc='upper right',
                        bbox_to_anchor=(legend_x_offset, 0.93),  # 只调整 x
                        frameon=False,
                        fontsize=26
                    )
                    for text in leg.get_texts():
                        text.set_color(STYLE['text_color'])
                        text.set_path_effects([pe.withStroke(linewidth=2, foreground=STYLE['bg_color'])])

                # --- 12. 保存 ---
                output_file = os.path.join(output_folder, f"{city_name}.png")
                plt.savefig(
                    output_file,
                    dpi=DPI,
                    facecolor=STYLE['bg_color'],
                    pad_inches=0
                )
                plt.close(fig)

            except Exception as e:
                print(f"  {city_name}: 错误")
                traceback.print_exc()


def plot_violin(indicator, input_folder=PATHS["RESULTS"], output_folder=PATHS["RESULTS"]):
    # 1. 数据读取与处理
    input_path = Path(input_folder) / indicator
    all_files = list(input_path.glob("*.csv"))

    if not all_files:
        print(f"Error: No CSV files found in {input_folder}")
        return

    print(f"Found {len(all_files)} files. Loading per city...")

    city_data = []

    for f in all_files:
        try:
            city_name = f.stem
            temp_df = pd.read_csv(f, usecols=['is_deployed', indicator])
            temp_df = temp_df.dropna(subset=[indicator, 'is_deployed'])

            # 确保两组都有数据
            if (temp_df['is_deployed'] == 1).any() and (temp_df['is_deployed'] == 0).any():
                city_data.append((city_name, temp_df))
            else:
                print(f"Skipping {city_name}: Missing data in one of the groups.")

        except ValueError:
            continue

    if not city_data:
        print("No valid data loaded.")
        return

    # 按数据量大小排序
    city_data.sort(key=lambda x: len(x[1]), reverse=True)

    total_cities = len(city_data)
    chunk_size = 9
    print(f"Total valid cities: {total_cities}. Generating plots in batches of {chunk_size}...")

    # 2. 分页循环绘图
    for page_idx, i in enumerate(range(0, total_cities, chunk_size)):

        # 获取当前页面的数据切片
        batch = city_data[i: i + chunk_size]
        current_page_num = page_idx + 1
        print(f"  Processing Page {current_page_num} (Cities {i + 1} to {i + len(batch)})...")

        # 创建 3x3 画布
        fig, axes = plt.subplots(3, 3, figsize=(16, 16), constrained_layout=True)
        axes_flat = axes.flatten()

        # 遍历 9 个子图位置
        for idx, ax in enumerate(axes_flat):
            if idx >= len(batch):
                ax.axis('off')
                continue

            city_name, df = batch[idx]

            deployed_vals = df[df['is_deployed'] == 1][indicator].values
            non_deployed_vals = df[df['is_deployed'] == 0][indicator].values

            # --- 统计检验 (Mann-Whitney U) ---
            try:
                _, p = mannwhitneyu(deployed_vals, non_deployed_vals, alternative='two-sided')
                p_text = "P < 0.01" if p < 0.01 else f"P = {p:.3f}"
                if p >= 0.1: p_text = "n.s."
            except ValueError:
                p_text = "Error"

            # --- 绘图 ---
            parts = ax.violinplot([deployed_vals, non_deployed_vals], positions=[1, 2], showextrema=False, widths=0.8)

            for pc, color in zip(parts['bodies'], ['lightblue', 'lightcoral']):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')

            for k, vals in enumerate([deployed_vals, non_deployed_vals], 1):
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                ax.errorbar(k, med, yerr=[[med - q1], [q3 - med]], fmt='o', c='k', lw=2, capsize=4, zorder=5)

            ax.set_title(f"{city_name}\n({p_text})", fontweight='bold')
            ax.set_xticks([1, 2])

            # 智能 X 轴标签逻辑：
            # 如果是最后一行 (idx >= 6) 或者 该列下方没有图了 (idx + 3 >= len(batch))
            # 则显示标签，否则隐藏
            if idx >= 6 or (idx + 3 >= len(batch)):
                ax.set_xticklabels(["AV-served", "AV-unserved"], fontsize=20)
            else:
                ax.set_xticklabels([])

            # Y轴科学计数法
            sci_indicators = []
            if indicator in sci_indicators:
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(formatter)

        # 3. 全局标签
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        fig.supylabel(LABELS.get(indicator, indicator))

        # 4. 保存当前页
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        # 文件名格式: indicator_p1.png
        save_name = f"{indicator}_p{current_page_num}.png"
        plt.savefig(out_path / save_name, dpi=300)
        plt.close()


def calculate_hedges_g_se(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)

    # Require at least 2 observations per group for variance estimation
    if nx < 2 or ny < 2:
        return np.nan, np.nan

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))

    # Avoid division by zero (with tolerance for floating-point precision)
    if pooled_sd < 1e-12:
        d = 0.0
        var_d = np.inf
    else:
        d = (mean_x - mean_y) / pooled_sd
        # Variance of Cohen's d (Hedges & Olkin, 1985 approximation)
        var_d = ((nx + ny) / (nx * ny)) + (d ** 2) / (2 * (nx + ny - 2))

    # Hedges' correction factor (J)
    J = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    g = d * J

    # Variance and SE of Hedges' g
    var_g = (J ** 2) * var_d
    se = np.sqrt(var_g) if np.isfinite(var_g) and var_g >= 0 else np.nan

    return g, se


def random_effects_meta_analysis(g_vals, se_vals):
    g = np.array(g_vals)
    se = np.array(se_vals)
    v = se ** 2

    # Fixed-effect weights
    w_fixed = 1 / v
    w_sum = np.sum(w_fixed)

    # Prevent division by zero
    if w_sum == 0:
        return np.nan, np.nan, np.nan, np.nan

    g_bar_fe = np.sum(w_fixed * g) / w_sum

    # Q statistic
    Q = np.sum(w_fixed * (g - g_bar_fe) ** 2)
    df = len(g) - 1

    if df <= 0:
        return g[0], g[0] - 1.96 * se[0], g[0] + 1.96 * se[0], 0.0

    # Estimate tau^2 (DerSimonian-Laird)
    denominator = w_sum - (np.sum(w_fixed ** 2) / w_sum)
    if denominator <= 1e-9:
        tau2 = 0.0
    else:
        tau2 = max(0, (Q - df) / denominator)

    # Random-effects weights
    w_random = 1 / (v + tau2)
    w_random_sum = np.sum(w_random)

    if w_random_sum == 0:
        return np.nan, np.nan, np.nan, tau2

    g_bar_re = np.sum(w_random * g) / w_random_sum
    var_g_bar_re = 1 / w_random_sum
    se_g_bar_re = np.sqrt(var_g_bar_re)

    ci_low = g_bar_re - 1.96 * se_g_bar_re
    ci_high = g_bar_re + 1.96 * se_g_bar_re

    return g_bar_re, ci_low, ci_high, tau2


def plot_cohens_d(indicators, input_folder=PATHS["RESULTS"], LABELS=LABELS):
    summary_data = []
    raw_points = []
    meta_results_list = []

    # --- 数据处理部分 ---
    for ind in indicators:
        input_path = Path(input_folder) / ind
        if not input_path.exists():
            print(f"Warning: Path not found {input_path}")
            continue

        all_files = list(input_path.glob("*.csv"))
        city_g_values = []
        city_se_values = []

        for f in all_files:
            try:
                df = pd.read_csv(f)
                if 'is_deployed' not in df.columns or ind not in df.columns:
                    continue
                df = df[['is_deployed', ind]].dropna()
                group_deployed = df[df['is_deployed'] == 1][ind].values
                group_non = df[df['is_deployed'] == 0][ind].values
                g, se = calculate_hedges_g_se(group_deployed, group_non)
                if not np.isnan(g) and not np.isnan(se) and se > 1e-8:
                    city_g_values.append(g)
                    city_se_values.append(se)
            except Exception:
                continue

        if city_g_values:
            g_overall, ci_low, ci_high, tau2 = random_effects_meta_analysis(city_g_values, city_se_values)
            meta_data = {
                'Indicator': ind,
                'Hedges_g': g_overall,
                'CI_low': ci_low,
                'CI_high': ci_high,
                'tau2': tau2,
                'n_cities': len(city_g_values)
            }
            meta_results_list.append(meta_data)
            summary_data.append(meta_data)
            raw_points.append(city_g_values)
        else:
            print(f"Warning: No valid data found for {ind}")
            raw_points.append([])

    if not summary_data:
        print("No data to plot.")
        return pd.DataFrame()

    # --- 绘图设置 ---
    # 颜色定义
    COLOR_POS = '#d62728'  # 鲜艳红 (元分析)
    COLOR_NEG = '#1f77b4'  # 鲜艳蓝 (元分析)
    SCATTER_POS = '#ff9896'  # 浅红 (散点)
    SCATTER_NEG = '#aec7e8'  # 浅蓝 (散点)
    BOX_FILL = '#f0f0f0'
    BOX_EDGE = '#555555'

    fig, ax = plt.subplots(figsize=(18, 8))

    raw_labels = [item['Indicator'] for item in summary_data]
    display_labels = [f"{LABELS_UNITLESS.get(label, label)}"
                      for label, item in zip(raw_labels, summary_data)]
    x_pos = np.arange(len(raw_labels))

    # A. 绘制箱线图
    bplot = ax.boxplot(raw_points, positions=x_pos, widths=0.5, patch_artist=True,
                       showfliers=False, zorder=1,
                       boxprops=dict(alpha=1.0, color=BOX_EDGE, linewidth=1.2),
                       whiskerprops=dict(color=BOX_EDGE, alpha=1.0, linewidth=1.2),
                       capprops=dict(color=BOX_EDGE, alpha=1.0, linewidth=1.2),
                       medianprops=dict(color='#333333', alpha=1.0, linewidth=1.5))

    for patch in bplot['boxes']:
        patch.set_facecolor(BOX_FILL)
        patch.set_edgecolor(BOX_EDGE)
        patch.set_alpha(0.7)

    # B. 绘制散点 (带抖动)
    np.random.seed(42)
    for i, points in enumerate(raw_points):
        if not points:
            continue
        jitter_x = np.random.normal(x_pos[i], 0.05, size=len(points))
        point_colors = [SCATTER_POS if p >= 0 else SCATTER_NEG for p in points]
        ax.scatter(jitter_x, points, s=80, alpha=0.6, c=point_colors,
                   zorder=2, edgecolors='white', linewidth=0.8)

    # C. 绘制元分析结果
    meta_means = [item['Hedges_g'] for item in summary_data]
    err_lows = [item['Hedges_g'] - item['CI_low'] for item in summary_data]
    err_highs = [item['CI_high'] - item['Hedges_g'] for item in summary_data]

    meta_colors = [COLOR_POS if m >= 0 else COLOR_NEG for m in meta_means]

    for i, (x, mean, elow, ehigh, color) in enumerate(zip(x_pos, meta_means, err_lows, err_highs, meta_colors)):
        # 绘制误差棒
        ax.errorbar(x, mean,
                    yerr=[[elow], [ehigh]],
                    fmt='none',
                    ecolor=color,
                    elinewidth=3.5, capsize=8, capthick=3.5,
                    zorder=10)
        # 单独绘制菱形点，加个黑色描边使其更突出
        ax.plot(x, mean, marker='D', markersize=12, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=2.5, zorder=11)

    # D. 辅助线
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    # 虚线参考线
    # ref_configs = [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]
    # for val, label in ref_configs:
    #     ax.axhline(val, color='gray', linestyle='--', linewidth=1, alpha=0.4, zorder=0)
    #     ax.axhline(-val, color='gray', linestyle='--', linewidth=1, alpha=0.4, zorder=0)
    #     ax.text(x_pos[-1] + 0.6, val, label, va='center', color='gray')

    # E. 坐标轴设置
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, rotation=30, ha='center', fontsize=24)

    ax.set_ylabel("Hedges' g", fontsize=24)
    ax.tick_params(axis='y', labelsize=20)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)

    # --- F. 图例 ---
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='black',
               markersize=10, markeredgewidth=3, label='Weighted mean'),
        Line2D([0], [0], color=COLOR_POS, lw=4, label='95% CI (Positive)'),
        Line2D([0], [0], color=COLOR_NEG, lw=4, label='95% CI (Negative)')
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper center',          # 位置锚点：顶部居中
        bbox_to_anchor=(0.5, 0.98),  # 坐标：X居中(0.5), Y接近顶部(0.98)
        ncol=3,                      # 3列横向排列
        frameon=True,                # 显示边框
        facecolor='white',           # 背景白色
        edgecolor='gray',            # 边框灰色
        framealpha=1,                # 不透明度 100%
        fancybox=False,              # 直角边框 (若喜欢圆角改为 True)
        fontsize=20                  # 字体大小
    )

    plt.tight_layout()

    out_path = Path(input_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / "Meta_Analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    return pd.DataFrame(meta_results_list)


def plot_vif(csv_path, output_dir=PATHS["RESULTS"]):
    # 1. 读取数据
    if not os.path.exists(csv_path):
        print(f"Error: VIF file {csv_path} not found.")
        return

    vif_df = pd.read_csv(csv_path)

    # 2. 数据预处理
    # 将变量名映射为可读标签
    vif_df['Label'] = vif_df['Variable'].map(LABELS_UNITLESS).fillna(vif_df['Variable'])
    # 按 VIF 大小排序 (barh 是从下往上画，所以从小到大排序，大的在上面)
    vif_df = vif_df.sort_values('VIF', ascending=True)

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = []
    for v in vif_df['VIF']:
        if pd.isna(v):
            colors.append('gray')
        elif v < 5:
            colors.append('#2ecc71')  # Green
        elif v < 10:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    # 使用 Label 作为 Y 轴数据
    bars = ax.barh(vif_df['Label'], vif_df['VIF'], color=colors, alpha=0.8, edgecolor='none', height=0.6)

    # Reference lines
    ax.axvline(x=10, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add labels for thresholds — 增大字体，加粗
    trans = ax.get_xaxis_transform()
    ax.text(10, 1.01, 'Threshold=10', transform=trans, color='#e74c3c', ha='center', weight='bold', fontsize=14)

    # X轴标签 — 设置字体大小
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=16, labelpad=10, fontweight='medium')

    # Y轴标签（指标名称）— 设置字体更大、加粗
    # barh 自动处理了 Y 轴刻度位置，这里只需要设置标签样式
    ax.set_yticks(range(len(vif_df)))
    ax.set_yticklabels(vif_df['Label'], fontsize=16, rotation=0)

    # Clean up spines — 显示所有边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    ax.grid(axis='x', linestyle=':', alpha=0.6)

    # Adjust x-limit
    max_vif = vif_df['VIF'].max()
    if pd.notna(max_vif):
        ax.set_xlim(0, max(max_vif * 1.15, 12))

    # Annotate bars
    for bar, vif in zip(bars, vif_df['VIF']):
        width = bar.get_width()
        if pd.notna(vif):
            label_x_pos = width + (ax.get_xlim()[1] * 0.01)
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                     f'{vif:.1f}', va='center', fontsize=14, color='#333333', fontweight='bold')

    plt.tight_layout()
    vif_path = os.path.join(output_dir, "vif_plot.png")
    fig.savefig(vif_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def plot_forest(csv_path, output_dir=PATHS["RESULTS"]):
    # 1. 读取数据
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df = df[df['term'] != 'spatial_lag'].copy()

    df['Label'] = df['term'].map(LABELS_UNITLESS).fillna(df['term'])

    # 2. 数据排序 (按 OR 大小排序)
    df = df.sort_values('OR').reset_index(drop=True)

    # 3. 设置画布
    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(df))

    # 4. 定义颜色逻辑
    colors = ['#d62728' if x > 1 else '#1f77b4' for x in df['OR']]

    # 5. 绘制背景条纹
    for i in range(len(df)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='gray', alpha=0.1, zorder=0)

    # 6. 绘制基准线
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

    # 7. 绘制误差棒和点
    for i, (idx, row) in enumerate(df.iterrows()):
        # 绘制误差线 (CI)
        ax.plot([row['Lower_CI'], row['Upper_CI']], [i, i],
                color=colors[i], linewidth=2, alpha=0.6, zorder=2)

        # 绘制误差线端点 (Caps)
        ax.plot([row['Lower_CI'], row['Lower_CI']], [i - 0.1, i + 0.1],
                color=colors[i], linewidth=1.5, alpha=0.6, zorder=2)
        ax.plot([row['Upper_CI'], row['Upper_CI']], [i - 0.1, i + 0.1],
                color=colors[i], linewidth=1.5, alpha=0.6, zorder=2)

        # 绘制中心点 (OR)
        ax.scatter(row['OR'], i, color=colors[i], s=100, zorder=3, edgecolors='white')

        # 8. 添加文字标注
        p = row['p.value']
        sig_text = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        or_text = f"{row['OR']:.2f}"

        t = ax.text(row['OR'], i + 0.15, or_text,
                    ha='center', va='bottom',
                    fontweight='bold', color=colors[i], fontsize=20)

        if sig_text:
            ax.annotate(sig_text,
                        xy=(1, 0), xycoords=t,  # 锚点设在 OR 值的右下角
                        xytext=(2, -7), textcoords='offset points',
                        ha='left', va='bottom',  # 星号左对齐
                        fontweight='bold', color=colors[i], fontsize=24)

    # 9. 轴设置
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Label'])

    # 设置 X 轴为对数刻度
    ax.set_xscale('log')

    # 自定义 X 轴刻度 (确保包含 2 和 5，覆盖更广的范围)
    xticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    ax.set_xticks(xticks)

    # 强制不使用科学计数法 (例如显示 0.1 而不是 10^-1)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticklabels([str(x) for x in xticks])

    # 计算并设置 X 轴范围 (增加 Padding)
    min_ci = df['Lower_CI'].min()
    max_ci = df['Upper_CI'].max()

    # 防止 min_ci 为 0 或负数导致 log 报错
    if min_ci <= 0: min_ci = 0.01

    ax.set_xlim(min_ci * 0.5, max_ci * 2.5)
    ax.set_ylim(-0.6, len(df) - 1 + 0.6)

    ax.set_xlabel("Odds ratio")

    plt.tight_layout()

    # 保存
    output_file = os.path.join(output_dir, 'forest_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_morans_i_comparison(morans_main_path, morans_robust_path, output_folder=PATHS["RESULTS"]):
    # --- 1. 读取数据 ---
    try:
        df_main = pd.read_csv(morans_main_path)
        df_robust = pd.read_csv(morans_robust_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 提取数值数组 (移除 NaN)
    vals_main = df_main['moran_i'].dropna().values
    vals_robust = df_robust['moran_i'].dropna().values

    print(f"Samples: Main={len(vals_main)}, Robustness={len(vals_robust)}")

    # --- 2. 创建画布 ---
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_axes([0.20, 0.12, 0.75, 0.82])

    # 背景与网格
    ax.set_facecolor('#eaeaea')
    ax.grid(axis='y', linestyle='-', alpha=1, color='white', linewidth=1.5, zorder=0)

    colors = ['#1f4e79', '#d66a54']

    # --- 3. 绘制抖动散点 (Jitter Plot) ---
    jitter_strength = 0.08
    for i, vals in enumerate([vals_main, vals_robust]):
        x = np.random.normal(i + 1, jitter_strength, size=len(vals))
        ax.scatter(x, vals,
                   alpha=0.6,       # 透明度
                   s=120,           # 点大小
                   color=colors[i], # 对应颜色
                   edgecolor='none',
                   zorder=2)        # 层级在箱线图之下

    # --- 4. 绘制箱线图 ---
    bplot = ax.boxplot([vals_main, vals_robust],
                       positions=[1, 2],
                       widths=0.5,
                       patch_artist=True,
                       showfliers=False,  # 隐藏离群值 (参考风格)
                       notch=True,
                       zorder=3)

    # 设置箱体样式
    for patch in bplot['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        patch.set_alpha(1)

    # 设置须、帽、中位线样式
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bplot[element], color='black', linewidth=2.5)

    # 中位线稍微细一点
    plt.setp(bplot['medians'], linewidth=2)

    # --- 5. 绘制均值点 ---
    means = [np.mean(vals_main), np.mean(vals_robust)]
    ax.scatter([1, 2], means, color='#333333', s=480, zorder=4, edgecolor='white', linewidth=2)

    # --- 6. 设置 Y 轴范围 ---
    whisker_data = [item.get_ydata() for item in bplot['whiskers']]
    all_whisker_vals = [val for sublist in whisker_data for val in sublist]

    if all_whisker_vals:
        y_min_vis = min(all_whisker_vals)
        y_max_vis = max(all_whisker_vals)
    else:
        y_min_vis = min(np.min(vals_main), np.min(vals_robust))
        y_max_vis = max(np.max(vals_main), np.max(vals_robust))

    y_range_vis = y_max_vis - y_min_vis
    if y_range_vis == 0: y_range_vis = 1

    # 添加内边距: 上下各留 5%
    y_min_plot = y_min_vis - y_range_vis * 0.05
    y_max_plot = y_max_vis + y_range_vis * 0.05

    ax.set_ylim(y_min_plot, y_max_plot)

    # 添加 0 线 (Moran's I 的基准线)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)

    # --- 7. 标签与刻度 ---
    ax.set_ylabel("Moran's I (Residuals)", labelpad=10)

    # 设置 X 轴标签
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Main model', 'Robustness model'])
    ax.set_xlim(0.7, 2.3)

    # 优化 Y 轴刻度显示
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))

    # --- 8. 保存图片 ---
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    save_full_path = out_path / "morans_i_comparison.png"
    plt.savefig(save_full_path, dpi=300)
    plt.close()


def plot_roc_comparison(predictions_path, output_folder=PATHS["RESULTS"]):
    df = pd.read_csv(predictions_path, header=None)
    df.columns = ['city', 'grid_id', 'y_true', 'pred_MAIN', 'pred_ROBUST']

    df = df.dropna(subset=['y_true', 'pred_MAIN', 'pred_ROBUST'])
    df['y_true'] = pd.to_numeric(df['y_true'], errors='coerce')
    df = df.dropna(subset=['y_true'])
    df['y_true'] = df['y_true'].astype(int)

    unique_labels = df['y_true'].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"y_true contains non-binary values: {unique_labels}")
    if len(unique_labels) < 2:
        raise ValueError("y_true has only one class; ROC requires both 0 and 1.")

    y_true = df['y_true'].values.astype(float)
    y_main = df['pred_MAIN'].values.astype(float)
    y_robust = df['pred_ROBUST'].values.astype(float)

    # 计算 ROC 和 AUC
    fpr_main, tpr_main, _ = roc_curve(y_true, y_main)
    fpr_robust, tpr_robust, _ = roc_curve(y_true, y_robust)
    auc_main = auc(fpr_main, tpr_main)
    auc_robust = auc(fpr_robust, tpr_robust)

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.20, 0.12, 0.75, 0.80])

    # 绘制 ROC 曲线（线宽=3）
    line_main, = ax.plot(fpr_main, tpr_main, color='red', linewidth=3)
    line_robust, = ax.plot(fpr_robust, tpr_robust, color='blue', linewidth=3)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8)

    # 坐标轴
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # --- 图例 ---
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label=f'Main model (AUC = {auc_main:.2f})'),
        Line2D([0], [0], color='blue', lw=3, label=f'Robustness model (AUC = {auc_robust:.2f})')
    ]

    ax.legend(
        handles=legend_elements,
        loc='lower right',           # 锚点为图例的左下角
        bbox_to_anchor=(1, 0), # X=0.98（贴近右边界），Y=0.01（贴近底边）
        ncol=1,                      # 右下角空间有限，建议单列
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=1,
        fancybox=False
    )

    # 保存
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    save_name = "roc_comparison.png"
    plt.savefig(out_path / save_name, dpi=300)
    plt.close()


# ==========================================
# Main 函数
# ==========================================
def main(indicator):
    print(f"\n=== Analysis Task: {indicator} ===")

    tasks = {
        "grid": lambda acc: run_grid_generation(acc),
        "ntl_mean": lambda acc: compute_ntl_mean(PATHS["NTL_TIF"], "ntl_mean"),
        "slope": lambda acc: compute_raster_mean(PATHS["SLOPE_TIF"], "slope"),
        "entropy": lambda acc: compute_road_entropy("entropy"),
        "road_density": lambda acc: compute_road_density("road_density"),
        "sinuosity": lambda acc: compute_sinuosity("sinuosity"),
        "ratio_complex": lambda acc: compute_ratio_complex("ratio_complex"),
        "dist_to_station": lambda acc: compute_station_distance("dist_to_station"),
        "poi_richness": lambda acc: compute_poi_richness("poi_richness")
    }

    if indicator in tasks:
        print(f"--- Running {indicator} ---")
        tasks[indicator](None)
        print("--- Plotting ---")
        plot_violin(indicator)
    else:
        print(f"未定义的指标或任务: {indicator}")
        print(f"可用任务: {list(tasks.keys())}")


if __name__ == "__main__":
    if "correlation_matrix_MAIN.csv" in os.listdir(PATHS["RESULTS"]):
        # 4. 画诊断图 (运行Intra_City_9.R之后)
        print(">>> Generating Diagnostic Plots...")

        # 1. Meta Analysis / Cohen's d
        plot_cohens_d(indicators=['ntl_mean_log', 'slope', 'road_density', 'sinuosity',
                                  'ratio_complex', 'dist_to_station', 'poi_richness_log'])

        # 2. VIF Plot
        plot_vif(csv_path=PATHS["RESULTS"] / "vif_results_MAIN.csv")

        # 3. Forest Plot
        plot_forest(PATHS["RESULTS"] / "glmm_results_ROBUST.csv")

        # 4. Moran's I Comparison
        plot_morans_i_comparison(morans_main_path=PATHS["RESULTS"] / "morans_i_results_MAIN.csv",
                                 morans_robust_path=PATHS["RESULTS"] / "morans_i_results_ROBUST.csv")

        # 5. ROC Curve
        plot_roc_comparison(predictions_path=PATHS["RESULTS"] / "predictions_for_ROC.csv")

    else:
        # 1. 生成网格
        main(indicator="grid")

        # 2. 生成叠加图
        generate_overlay_maps("ntl_mean", PATHS["GRID_OUTPUT"], PATHS["NTL_TIF"], PATHS["OVERLAY_OUTPUT"])

        # 3. 指标比较
        main(indicator="ntl_mean")
        main(indicator="slope")
        main(indicator="entropy")
        main(indicator="road_density")
        main(indicator="sinuosity")
        main(indicator="ratio_complex")
        main(indicator="dist_to_station")
        main(indicator="poi_richness")