import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import scienceplots
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim
import rasterio
from utilities.calc_gdp_per_capita import calc_gdp_per_capita
from utilities.extract_citi_names import extract_citi_names


# Hyperparameters of scienceplots
plt.style.use(['science', 'no-latex', 'nature'])

plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'legend.title_fontsize': 20,
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


# 气候类型映射字典
CLIMATE_MAPPING = {
    1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk',
    8: 'Csa', 9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
    14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
    20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa',
    26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 29: 'ET', 30: 'EF'
}


def read_gmt_border(file_path):
    polygons = []
    current = []
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: China border file not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if current:
                    polygons.append(np.array(current))
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        current.append([lon, lat])
                    except ValueError:
                        continue
        if current:
            polygons.append(np.array(current))
    return polygons


def geocode_cities(cities):
    geolocator = Nominatim(user_agent="city_map_plotter_v2")
    coords = {}
    countries = {}

    for city in cities:
        try:
            # 针对三亚的特殊处理，确保能搜到
            search_query = city

            loc = geolocator.geocode(search_query, timeout=10)
            if loc:
                save_key = "Sanya" if city == "Sanya, China" else city

                coords[save_key] = (loc.latitude, loc.longitude)
                print(f"✅ Geocoded {city} -> {save_key}: ({loc.latitude:.4f}, {loc.longitude:.4f})")

                display_name = loc.raw.get('display_name', '')
                parts = [part.strip() for part in display_name.split(',')][-1]

                if parts == "中国":
                    countries[save_key] = "China"
                elif parts == "Deutschland":
                    countries[save_key] = "Germany"
                elif parts == "대한민국":
                    countries[save_key] = "Korea"
                elif parts == "الإمارات العربية المتحدة":
                    countries[save_key] = "UAE"
                elif parts == "United States":
                    countries[save_key] = "USA"
                else:
                    countries[save_key] = parts
            else:
                print(f"⚠️  Not found {city}")
        except Exception as e:
            print(f"⚠️  Error geocoding {city}: {e}")
    return coords, countries


def map_gdp_to_sizes(gdp_values, min_size=20, max_size=180, power=0.5):
    gdp_array = np.array(gdp_values)
    gdp_array = np.clip(gdp_array, 1e-3, None)
    powered = gdp_array ** power
    sizes = np.interp(powered, (powered.min(), powered.max()), (min_size, max_size))
    return sizes


def extract_climate_from_tif(tif_path, lat, lon):
    try:
        with rasterio.open(tif_path) as src:
            if (lon < src.bounds.left or lon > src.bounds.right or
                    lat < src.bounds.bottom or lat > src.bounds.top):
                return None
            row, col = src.index(lon, lat)
            value = src.read(1)[row, col]
            return int(value)
    except Exception as e:
        # print(f"⚠️ Error reading TIF for {lat}, {lon}: {e}")
        return None


# ========== Plot main map ==========
def plot_main_map(valid_lons, valid_lats, valid_counts, valid_gdp, climate_labels,
                  sizes, discrete_cmap, norm, china_gmt_file, output_file):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    land_color = '#f0f0f0'
    ocean_color = '#e8e8e8'
    border_color = '#aaaaaa'

    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=border_color, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.6, edgecolor=border_color, zorder=1)

    # Draw China border
    china_polys = read_gmt_border(china_gmt_file)
    for poly in china_polys:
        if len(poly) > 2:
            polygon_fill = MplPolygon(
                poly,
                facecolor=land_color,
                edgecolor=border_color,
                linewidth=0.6,
                linestyle='-',
                transform=ccrs.PlateCarree(),
                zorder=3
            )
            ax.add_patch(polygon_fill)

    # Draw city points
    scatter = ax.scatter(valid_lons, valid_lats,
                         c=valid_counts,
                         s=sizes,
                         cmap=discrete_cmap,
                         norm=norm,
                         edgecolor='black',
                         linewidth=0.6,
                         transform=ccrs.PlateCarree(),
                         zorder=10,
                         alpha=0.8)

    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    # 绘制直方图
    if climate_labels:
        climate_series = pd.Series(climate_labels)
        counts = climate_series.value_counts()

        x_labels = counts.index.astype(str).tolist()
        y_values = counts.values
        x_pos = np.arange(len(x_labels))

        ax_hist = ax.inset_axes([0.375, 0.07, 0.35, 0.15], transform=ax.transAxes)
        ax_hist.xaxis.set_major_locator(plt.NullLocator())
        ax_hist.xaxis.set_major_formatter(plt.NullFormatter())

        # 颜色、边框、透明度与 reference 保持一致
        bars = ax_hist.bar(x_pos, y_values,
                           color='#d62728',  # Reference color
                           edgecolor='white',  # Reference edge color
                           linewidth=0.5,  # Reference linewidth
                           alpha=0.7)  # Reference alpha

        for spine in ax_hist.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')

        ax_hist.set_xticks(x_pos)
        ax_hist.set_xticklabels(x_labels, ha='center')

        ax_hist.set_title('Köppen climate types', fontsize=16, pad=5)
        ax_hist.tick_params(axis='x', labelsize=16)
        ax_hist.tick_params(axis='y', labelsize=16)
        ax_hist.set_ylabel('Count', fontsize=16)
        ax_hist.patch.set_alpha(0.8)

    # Color legend
    cax = ax.inset_axes([0.02, 0.32, 0.2, 0.03], transform=ax.transAxes)
    cbar = plt.colorbar(
        ScalarMappable(norm=norm, cmap=discrete_cmap),
        cax=cax,
        orientation='horizontal',
        ticks=[1, 2, 3, 4]
    )
    cbar.set_label('Number of AV service providers', fontsize=16, labelpad=8)
    cbar.ax.tick_params(labelsize=16)

    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor('black')

    # GDP size legend
    gdp_arr = np.array(valid_gdp)
    if len(gdp_arr) > 0:
        gdp_min_raw = np.min(gdp_arr)
        gdp_max_raw = np.max(gdp_arr)

        # 优化 Legend 显示逻辑，防止 min/max 过于接近
        gdp_min_rounded = max(5000, int(np.floor(gdp_min_raw / 5000) * 5000))
        gdp_max_rounded = int(np.ceil(gdp_max_raw / 5000) * 5000)

        if gdp_max_rounded - gdp_min_rounded < 15000:
            gdp_max_rounded = gdp_min_rounded + 20000

        gdp_legend_vals = np.linspace(gdp_min_rounded, gdp_max_rounded, 4)
        gdp_legend_vals = np.round(gdp_legend_vals).astype(int)
        gdp_legend_vals = (gdp_legend_vals // 5000) * 5000
        gdp_legend_vals = np.unique(gdp_legend_vals)

        # 确保至少有几个点
        if len(gdp_legend_vals) < 3:
            gdp_legend_vals = np.array([gdp_min_rounded, gdp_min_rounded + 10000, gdp_min_rounded + 20000])

        size_legend = map_gdp_to_sizes(gdp_legend_vals)

        legend_elements = []
        for gdp_val, size_val in zip(gdp_legend_vals, size_legend):
            label = f"{gdp_val // 10000:,}"
            legend_elements.append(
                plt.scatter([], [], s=size_val, c='gray', edgecolor='black',
                            linewidth=0.6, label=label)
            )

        # 水平放置图例，位于左下角
        legend = ax.legend(
            handles=legend_elements,
            title=r'GDP per capita ($10^4$ US\$)',
            loc='lower left',
            bbox_to_anchor=(0, 0.02),
            frameon=True,
            facecolor='white', edgecolor='gray',
            labelspacing=0.6, borderpad=0.6,
            ncol=len(legend_elements),
            columnspacing=1,
            handletextpad=0.2
        )
        legend.get_title().set_fontsize(16)
        for text in legend.get_texts():
            text.set_fontsize(16)

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Main map saved: {output_file}")


def plot_region_map(region_name, bounds, valid_lons, valid_lats, valid_counts, sizes, discrete_cmap, norm, output_dir):
    if region_name == "Yangtze River Delta":
        fig = plt.figure(figsize=(3, 3))
    else:
        fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    land_color = '#f0f0f0'
    ocean_color = '#e8e8e8'
    border_color = '#aaaaaa'

    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor=border_color, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor=border_color, zorder=1)

    lon_min, lon_max, lat_min, lat_max = bounds
    region_lons, region_lats, region_counts, region_sizes = [], [], [], []
    for lon, lat, count, size in zip(valid_lons, valid_lats, valid_counts, sizes):
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            region_lons.append(lon)
            region_lats.append(lat)
            region_counts.append(count)
            region_sizes.append(size)

    if region_lons:
        ax.scatter(region_lons, region_lats,
                   c=region_counts,
                   s=region_sizes,
                   cmap=discrete_cmap,
                   norm=norm,
                   edgecolor='black',
                   linewidth=0.8,
                   transform=ccrs.PlateCarree(),
                   zorder=10,
                   alpha=0.8)

    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    ax.set_title("")

    safe_name = region_name.replace(" ", "_")
    output_file = os.path.join(output_dir, f"{safe_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Regional map saved: {output_file}")


def main():
    directory = "src/Results/Total_Dist/Corp/"
    china_file = "Data/CHN_Map/CN-border-L1.gmt"
    output_dir = "src/Results/City_Dist/"
    city_geojson_dir = "src/Results/Total_Dist/Built_Up/"
    climate_tif_path = "Data/QGIS/Koppen/1991_2020/koppen_geiger_0p00833333.tif"

    os.makedirs(output_dir, exist_ok=True)

    city_counts = extract_citi_names(directory)

    # 1. 保存 Counts CSV
    with open(output_dir + "city_counts.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['City', 'Count'])
        for city, count in city_counts.items():
            writer.writerow([city, count])

    # 2. 获取或加载坐标（同时获取国家信息）
    country_map = {}  # 用于存储 City -> Country 的映射

    if os.path.exists(output_dir + 'city_geocodes.csv'):
        print("Loading existing geocodes...")
        coords_df = pd.read_csv(output_dir + 'city_geocodes.csv')
        # 注意：这里加载后的 Key 是 CSV 里的名字（例如 "Sanya"）
        coords_raw = coords_df.set_index('City')[["Latitude", "Longitude"]].T.to_dict('list')
        coords = {city: tuple(pos) for city, pos in coords_raw.items()}
        # 新增：从 CSV 加载国家映射
        country_map = coords_df.set_index('City')['Country'].to_dict()
    else:
        print("Geocoding cities...")
        coords, countries = geocode_cities(list(city_counts.keys()))
        # 新增：保存国家映射
        country_map = countries
        data = []
        # 注意：这里需要遍历 coords 的 keys，因为 geocode_cities 内部可能已经把 "Sanya, China" 改成了 "Sanya"
        # 为了对应原始 city_counts，我们做一次反向匹配或者直接保存 coords 的内容
        for city_key, (lat, lon) in coords.items():
            country = countries.get(city_key, "Unknown")
            data.append({
                'City': city_key,
                'Latitude': lat,
                'Longitude': lon,
                'Country': country
            })
        df = pd.DataFrame(data)
        df.to_csv(output_dir + 'city_geocodes.csv', index=False, encoding='utf-8')

    # 3. 计算 GDP
    city_gdp_per_capita = {}
    print("Calculating GDP...")

    for city in city_counts:
        # 统一使用 lookup_key 来查找 coords
        # 如果 city_counts 里是 "Sanya, China"，但 coords 里是 "Sanya"，直接查 city 会失败
        lookup_key = "Sanya" if city == "Sanya, China" else city

        if lookup_key not in coords:
            print(f"⚠️ Skipping GDP for {city} (Key mismatch or no coords: looked for '{lookup_key}')")
            continue

        # GeoJSON 文件名通常也是 "Sanya.geojson"
        geojson_path = os.path.join(city_geojson_dir, f"{lookup_key}.geojson")

        try:
            if not os.path.exists(geojson_path):
                print(f"⚠️ GeoJSON not found: {geojson_path}")
                continue

            val = calc_gdp_per_capita(POLY_GEOJSON=geojson_path)
            if isinstance(val, (pd.DataFrame, pd.Series)):
                val = val.iloc[0]

            # 存储时使用 lookup_key，保持与 coords 一致
            city_gdp_per_capita[lookup_key] = val
            print(f"✅ {lookup_key} GDP: {val:,.0f}")
        except Exception as e:
            print(f"❌ Error calculating GDP for {lookup_key}: {e}")
            continue

    # 4. 整合数据用于绘图和生成统计表格
    valid_lons, valid_lats, valid_counts, valid_gdp = [], [], [], []
    climate_labels = []

    # 新增：用于存储表格数据的列表
    summary_data = []

    print("Aggregating plot data and statistics...")
    for city, count in city_counts.items():
        lookup_key = "Sanya" if city == "Sanya, China" else city

        if lookup_key in coords and lookup_key in city_gdp_per_capita:
            gdp = city_gdp_per_capita[lookup_key]

            if gdp is not None and gdp > 0:
                lat, lon = coords[lookup_key]

                # 获取气候类型
                current_climate_label = "Unknown"
                if os.path.exists(climate_tif_path):
                    clim_code = extract_climate_from_tif(climate_tif_path, lat, lon)
                    if clim_code is not None:
                        label = CLIMATE_MAPPING.get(clim_code, str(clim_code))
                        climate_labels.append(label)
                        current_climate_label = label
                    else:
                        pass

                # 获取国家
                current_country = country_map.get(lookup_key, "Unknown")

                valid_lats.append(lat)
                valid_lons.append(lon)
                valid_counts.append(count)
                valid_gdp.append(gdp)

                # 新增：添加到统计列表
                summary_data.append({
                    'City': lookup_key,
                    'Country': current_country,
                    'Service_Providers': count,
                    'GDP_Per_Capita': gdp,
                    'Climate_Type': current_climate_label
                })

                if lookup_key == "Sanya":
                    print(f"🌟 SANYA ADDED TO PLOT DATA: GDP={gdp}, Count={count}, Lat={lat}")
        else:
            if "Sanya" in city:
                print(
                    f"⚠️ Sanya dropped. In Coords? {'Yes' if lookup_key in coords else 'No'}. In GDP? {'Yes' if lookup_key in city_gdp_per_capita else 'No'}")

    # 保存统计表格
    if summary_data:
        stats_df = pd.DataFrame(summary_data)
        stats_output_path = os.path.join(output_dir, "city_statistics.csv")
        stats_df.to_csv(stats_output_path, index=False, encoding='utf-8')
        print(f"✅ City statistics saved: {stats_output_path}")

    # 5. 绘图
    if not valid_lons:
        print("❌ No valid data to plot!")
        return

    counts_array = np.array(valid_counts)
    capped_counts = np.clip(counts_array, 1, 4)
    bounds_color = [0.5, 1.5, 2.5, 3.5, 4.5]
    base_cmap = plt.cm.Reds
    colors = [base_cmap(i) for i in np.linspace(0.3, 1.0, 4)]
    discrete_cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds_color, ncolors=4)

    sizes = map_gdp_to_sizes(valid_gdp)

    plot_main_map(
        valid_lons=valid_lons,
        valid_lats=valid_lats,
        valid_counts=capped_counts,
        valid_gdp=valid_gdp,
        climate_labels=climate_labels,
        sizes=sizes,
        discrete_cmap=discrete_cmap,
        norm=norm,
        china_gmt_file=china_file,
        output_file=os.path.join(output_dir, "Global_Dist.png")
    )

    regions = {"Yangtze River Delta": [118, 122, 29.5, 33],
               "Pearl River Delta": [112.5, 114.5, 22, 23.5]}

    for region_name, bounds in regions.items():
        plot_region_map(
            region_name=region_name,
            bounds=bounds,
            valid_lons=valid_lons,
            valid_lats=valid_lats,
            valid_counts=capped_counts,
            sizes=sizes,
            discrete_cmap=discrete_cmap,
            norm=norm,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
