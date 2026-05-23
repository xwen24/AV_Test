import os
import geopandas as gpd
import scienceplots
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, PercentFormatter, MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from utilities.hyperparameters import COUNTRY_COLOR
import warnings
warnings.filterwarnings("ignore")

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

output_dir = 'src/Results/Area_Calc/'
os.makedirs(output_dir, exist_ok=True)


# ==========================================
# 通用工具函数
# ==========================================
def clean_df(df):
    """
    清理 DataFrame 的列名和所有字符串列中的 BOM (\ufeff) 及常见乱码前缀（如 'Ôªø', 'ï»¿'）
    """
    # 1. 清理列名开头的 BOM
    df.columns = df.columns.str.replace(r'^\ufeff', '', regex=True).str.strip()

    # 2. 找出所有 object 类型（通常是字符串）的列
    str_cols = df.select_dtypes(include=['object']).columns

    # 3. 对每个字符串列进行清洗
    for col in str_cols:
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = (
                df.loc[mask, col]
                .astype(str)
                .str.replace('\ufeff', '', regex=False)
                .str.replace(r'^Ôªø|^ï»¿', '', regex=True)
            )
        # 只在字符串列中替换 'nan'
        df[col] = df[col].replace('nan', pd.NA)

    return df


def load_gpkg(path, layer=None):
    return clean_df(gpd.read_file(path, layer=layer))


# Toggle this to True to plot results after running once with False
plot_Area_Calc = True

if not plot_Area_Calc:
    # Store results for all cities
    results_list = []

    for city_name in os.listdir('src/Results/Total_Dist/Built_Up/'):
        if city_name.endswith('.geojson'):
            city_base_name = city_name.replace('.geojson', '')

            # 将 Xiongan 改为 Xiong'an
            if city_base_name == 'Xiongan':
                city_base_name = "Xiong'an"

            print(f"\nProcessing {city_base_name}...")

            try:
                # 1. Read city boundary (GeoJSON)
                polygons_gdf = gpd.read_file('src/Results/Total_Dist/Built_Up/' + city_name)

                # 2. Read GPKG layer (with population data)
                layer_name = 'GHS_UCDB_THEME_GHSL_GLOBE_R2024A'
                shape_layer_gdf = load_gpkg('Data/QGIS/GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg',
                                            layer=layer_name)

                print("Original CRS:")
                print("  Polygons:", polygons_gdf.crs)
                print("  Shape layer:", shape_layer_gdf.crs)

                # 3. Reproject both layers to equal-area projection (EPSG:6933) for accurate area calculation
                target_crs = 'EPSG:6933'
                polygons_gdf = polygons_gdf.to_crs(target_crs)
                shape_layer_gdf = shape_layer_gdf.to_crs(target_crs)

                # 4. Compute original cell area (for proportional allocation later)
                # This represents the TOTAL area of the urban center
                shape_layer_gdf['original_area'] = shape_layer_gdf.geometry.area

                # 5. Compute intersection (preserving attributes from shape_layer, including population)
                intersections = gpd.overlay(polygons_gdf, shape_layer_gdf, how='intersection')

                # Initialize result variables
                total_area_m2 = 0
                total_area_km2 = 0
                total_population = 0

                # New variables for totals (Denominator)
                total_builtup_area_m2 = 0
                total_builtup_pop = 0

                status = ''

                # If no intersection, record result and proceed to next city
                if intersections.empty:
                    print("No intersection area found with GHS-UCDB!")
                    status = 'no_builtup_intersection'

                    # Since fallback raster logic is removed, we set everything to 0
                    total_area_m2 = 0
                    total_area_km2 = 0
                    total_population = 0
                    total_builtup_area_m2 = 0
                    total_builtup_pop = 0

                    results_list.append({
                        'city': city_base_name,
                        'intersect_area_m2': total_area_m2,
                        'intersect_area_km2': total_area_km2,
                        'estimated_population': total_population,
                        'total_builtup_area_m2': total_builtup_area_m2,
                        'total_builtup_pop': total_builtup_pop,
                        'status': status
                    })
                else:
                    # 6. Calculate intersection area
                    intersections['intersect_area'] = intersections.geometry.area

                    # 7. Select population column for estimation year
                    pop_col = '\ufeffGH_POP_TOT_2020'
                    if pop_col not in intersections.columns:
                        # Try finding column without BOM or similar name
                        possible_cols = [c for c in intersections.columns if 'GH_POP_TOT_2020' in c]
                        if possible_cols:
                            pop_col = possible_cols[0]
                        else:
                            raise ValueError(
                                f"Population column '{pop_col}' not found! Available columns starting with 'P': "
                                f"{[col for col in intersections.columns if col.startswith('P')]}")

                    # 8. Allocate population proportionally by area
                    intersections = intersections[intersections['original_area'] > 0]
                    intersections['pop_fraction'] = intersections['intersect_area'] / intersections['original_area']
                    intersections['estimated_population'] = intersections[pop_col] * intersections['pop_fraction']

                    # 9. Aggregate total area and total population (Numerator)
                    total_area_m2 = intersections['intersect_area'].sum()
                    total_population = intersections['estimated_population'].sum()
                    total_area_km2 = total_area_m2 / 1_000_000

                    # 10. Calculate Totals (Denominator)
                    # SPECIAL HANDLING FOR CHONGQING
                    if city_base_name == 'Chongqing':
                        print("Applying special denominator logic for Chongqing...")
                        # Filter all polygons in the original shape layer that belong to Chongqing
                        # Assuming the column name is 'GC_UCN_MAI_2025' based on request
                        if 'GC_UCN_MAI_2025' in shape_layer_gdf.columns:
                            chongqing_ucs = shape_layer_gdf[shape_layer_gdf['GC_UCN_MAI_2025'] == 'Chongqing']
                            if not chongqing_ucs.empty:
                                total_builtup_area_m2 = chongqing_ucs['original_area'].sum()
                                total_builtup_pop = chongqing_ucs[pop_col].sum()
                            else:
                                print(
                                    "Warning: No entries found for GC_UCN_MAI_2025 == 'Chongqing'. Falling back to intersection logic.")
                                # Fallback logic (same as other cities below)
                                id_col = 'ID_UC_G0'
                                if id_col in intersections.columns:
                                    unique_ucs = intersections.drop_duplicates(subset=[id_col])
                                    total_builtup_area_m2 = unique_ucs['original_area'].sum()
                                    total_builtup_pop = unique_ucs[pop_col].sum()
                                else:
                                    total_builtup_area_m2 = intersections['original_area'].sum()
                                    total_builtup_pop = intersections[pop_col].sum()
                        else:
                            print("Warning: Column 'GC_UCN_MAI_2025' not found. Falling back to intersection logic.")
                            # Fallback logic
                            id_col = 'ID_UC_G0'
                            if id_col in intersections.columns:
                                unique_ucs = intersections.drop_duplicates(subset=[id_col])
                                total_builtup_area_m2 = unique_ucs['original_area'].sum()
                                total_builtup_pop = unique_ucs[pop_col].sum()
                            else:
                                total_builtup_area_m2 = intersections['original_area'].sum()
                                total_builtup_pop = intersections[pop_col].sum()
                    else:
                        # STANDARD LOGIC FOR OTHER CITIES
                        # We need to sum the original area/pop of the UNIQUE urban centers involved.
                        # Intersection might split one UC into multiple polygons, so we must deduplicate by ID.
                        id_col = 'ID_UC_G0'
                        if id_col in intersections.columns:
                            unique_ucs = intersections.drop_duplicates(subset=[id_col])
                            total_builtup_area_m2 = unique_ucs['original_area'].sum()
                            total_builtup_pop = unique_ucs[pop_col].sum()
                        else:
                            print(
                                f"Warning: ID column {id_col} not found. Using sum of all fragments (may be inaccurate if overlaps exist).")
                            # Fallback if ID missing (unlikely for GHS data)
                            total_builtup_area_m2 = intersections['original_area'].sum()
                            total_builtup_pop = intersections[pop_col].sum()

                    print(f"\n===== Intersection results for {city_base_name} with GHS-UCDB =====")
                    print(
                        f"Covered Area: {total_area_km2:,.2f} km² / Total UC Area: {total_builtup_area_m2 / 1e6:,.2f} km²")
                    print(f"Covered Pop: {total_population:,.0f} / Total UC Pop: {total_builtup_pop:,.0f}")

                    # Record result
                    results_list.append({
                        'city': city_base_name,
                        'intersect_area_m2': total_area_m2,
                        'intersect_area_km2': total_area_km2,
                        'estimated_population': total_population,
                        'total_builtup_area_m2': total_builtup_area_m2,
                        'total_builtup_pop': total_builtup_pop,
                        'status': 'success'
                    })

                # ==========================================
                # PLOTTING SECTION
                # ==========================================

                # Define Plot CRS as EPSG:4326 to show Longitude/Latitude on axes
                plot_crs = "EPSG:4326"

                # Prepare layers for plotting by reprojecting to Lat/Lon
                polygons_plot = polygons_gdf.to_crs(plot_crs)
                layer_plot = shape_layer_gdf.to_crs(plot_crs)

                if not intersections.empty:
                    intersections_plot = intersections.to_crs(plot_crs)
                else:
                    intersections_plot = gpd.GeoDataFrame(geometry=[], crs=plot_crs)

                # Determine Intelligent Plotting Extent
                # Calculate bounds of the union of city boundary and intersection to ensure everything fits
                if not intersections_plot.empty:
                    # Combine geometries to find total bounds
                    combined_geo = pd.concat([polygons_plot.geometry, intersections_plot.geometry])
                    minx, miny, maxx, maxy = combined_geo.total_bounds
                else:
                    minx, miny, maxx, maxy = polygons_plot.total_bounds

                # Add a 10% margin to the extent so it's not too tight or too zoomed out
                x_span = maxx - minx
                y_span = maxy - miny
                margin_x = x_span * 0.1
                margin_y = y_span * 0.1

                # Plotting
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))

                # Set intelligent limits
                ax.set_xlim(minx - margin_x, maxx + margin_x)
                ax.set_ylim(miny - margin_y, maxy + margin_y)

                # 1) Plot layers
                # Bottom: GPKG layer (Urban Centers)
                if layer_plot is not None:
                    layer_plot.plot(ax=ax, facecolor="#888888", edgecolor="#555555", alpha=0.7)

                # Middle: City boundary (AV Service Area) - THICKER BLUE LINE
                polygons_plot.boundary.plot(ax=ax, color="#1565c0", linewidth=4)

                # Top: Intersection area
                if not intersections_plot.empty:
                    intersections_plot.plot(ax=ax, facecolor="#e53935", edgecolor="#b71c1c", alpha=0.5)
                else:
                    print("Note: intersections is empty; only city boundary and layer will be shown.")

                # 2) Add OSM basemap (Contextily supports EPSG:4326 reprojection)
                try:
                    ctx.add_basemap(ax, crs=plot_crs, source=ctx.providers.CartoDB.Positron)
                except Exception as e:
                    print(f"Failed to add basemap (possibly network/proxy issue): {e}")

                # 3) Styling and Labels
                ax.set_box_aspect(1)

                # Add Axis Labels (Requested: Longitude/Latitude)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')

                # --- FIX: Reduce Tick Density and Format to 2 Decimal Places ---
                # Limit the number of ticks to a maximum of 4 to prevent crowding
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

                # Format the tick labels to exactly 2 decimal places
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                # 4) Add City Name (Top-Left, Bold, Clear)
                ax.text(0.03, 0.97, city_base_name,
                        transform=ax.transAxes,
                        fontsize=24,
                        fontweight='bold',
                        va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

                # 5) Add legend - THICKER BLUE LINE in Legend
                legend_elements = [
                    Patch(facecolor="#888888", edgecolor="#555555", label="Urban centers"),
                    # Increased linewidth to 4 to match the plot
                    Patch(facecolor='none', edgecolor='#1565c0', linewidth=4, label='AV service area'),
                ]

                if not intersections_plot.empty:
                    legend_elements.append(
                        Patch(facecolor='#e53935', edgecolor='#b71c1c', alpha=0.5,
                              label='Covered urban centers')
                    )

                ax.legend(handles=legend_elements, loc='upper right', prop={'weight': 'bold'})

                plt.tight_layout()

                # Save figure
                plot_filename = os.path.join(output_dir, f'{city_base_name}.png')
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"Figure saved: {plot_filename}")
                plt.close()

            except Exception as e:
                print(f"Error processing {city_base_name}: {str(e)}")
                import traceback

                traceback.print_exc()
                results_list.append({
                    'city': city_base_name,
                    'intersect_area_m2': 0,
                    'intersect_area_km2': 0,
                    'estimated_population': 0,
                    'total_builtup_area_m2': 0,
                    'total_builtup_pop': 0,
                    'status': f'error: {str(e)}'
                })

    # Save results to CSV
    results_df = pd.DataFrame(results_list)
    csv_filename = os.path.join(output_dir, 'Area_Calc.csv')
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\nResults for all cities saved to: {csv_filename}")

    print(f"\n=== Processing Summary ===")
    print(f"Total cities processed: {len(results_list)}")
    print(f"Successfully processed cities: {len([r for r in results_list if r['status'] == 'success'])}")
    print(
        f"Cities with no built-up intersection: {len([r for r in results_list if r['status'] == 'no_builtup_intersection'])}")
    print(f"Cities with processing errors: {len([r for r in results_list if r['status'].startswith('error')])}")
else:
    # 1) Simple cache to avoid repeated geocoding
    cache_path = "city_country_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            CITY_COUNTRY_CACHE = json.load(f)
    else:
        CITY_COUNTRY_CACHE = {}


    def normalize_city(name: str) -> str:
        return name.strip().lower()


    # 2) Initialize Nominatim
    geolocator = Nominatim(user_agent="your-app-name-city-country", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)


    def infer_country_from_city(city_name: str) -> str:
        key = normalize_city(city_name)
        if key in CITY_COUNTRY_CACHE:
            return CITY_COUNTRY_CACHE[key]
        # Query directly by city name; if ambiguous, you may add ", city" or known region hints here
        loc = geocode(city_name, addressdetails=True)
        country = "Unknown"
        if loc and hasattr(loc, "raw"):
            addr = loc.raw.get("address", {})
            if "country" in addr:
                country = addr["country"]
            elif "country_code" in addr:
                country = addr["country_code"].upper()
        CITY_COUNTRY_CACHE[key] = country
        # Save cache immediately to prevent data loss if interrupted
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(CITY_COUNTRY_CACHE, f, ensure_ascii=False, indent=2)
        return country


    # Load Data
    csv_path = os.path.join(output_dir, "Area_Calc.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run the script with plot_Area_Calc = False first to generate the data.")
        exit(1)

    df = pd.read_csv(csv_path)
    df.set_index('city', inplace=True)

    # Extract Data for Plotting
    cities = list(df.index)
    covered_areas = [0 if pd.isna(x) else x for x in df['intersect_area_km2']]
    covered_populations = [0 if pd.isna(x) else x for x in df['estimated_population']]

    # New Data for Ratio Plotting
    # Avoid division by zero by replacing 0 with nan temporarily or handling it
    total_areas_m2 = df['total_builtup_area_m2'].fillna(0).values
    total_pops = df['total_builtup_pop'].fillna(0).values

    # Calculate Ratios (Handle division by zero)
    # Convert m2 to m2 for ratio (units cancel out)
    intersect_areas_m2 = df['intersect_area_m2'].fillna(0).values

    area_ratios = np.divide(intersect_areas_m2, total_areas_m2, out=np.zeros_like(intersect_areas_m2),
                            where=total_areas_m2 != 0) * 100
    pop_ratios = np.divide(covered_populations, total_pops, out=np.zeros_like(covered_populations),
                           where=total_pops != 0) * 100

    countries = []
    for city in cities:
        if city == "Sanya":
            temp_city_name = "Sanya, China"
        elif city == "Yangquan":
            temp_city_name = "Yangquan, China"
        else:
            temp_city_name = city
        country = infer_country_from_city(temp_city_name)
        countries.append(country)

    # ---------------------------------------------------------
    # PLOT 1: Absolute Values
    # ---------------------------------------------------------

    # Sort by covered area
    sorted_pairs = sorted(zip(cities, covered_areas, covered_populations, countries), key=lambda x: x[1])
    cities_sorted, covered_areas_sorted, covered_populations_sorted, countries_sorted = zip(*sorted_pairs)

    x = np.arange(len(cities_sorted))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(22, 8))

    # Add alternating background stripes
    for i in range(len(cities_sorted)):
        if i % 2 == 0:
            ax1.axvspan(i - 0.5, i + 0.5, facecolor='gray', alpha=0.3, zorder=0)


    class ZeroAwareScalarFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            if np.isclose(x, 0, atol=1e-12):
                return "0"
            return super().__call__(x, pos)


    # Left Y-axis: covered urban area (blue)
    bars1 = ax1.bar(x - width / 2, covered_areas_sorted, width, color='#4C72B0', label='Covered urban area (km²)')
    ax1.set_ylabel('Covered urban area (km²)', color='#4C72B0')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')

    formatter = ZeroAwareScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax1.yaxis.set_major_formatter(formatter)

    # Right Y-axis: covered population (red)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, covered_populations_sorted, width, color='#C44E52', label='Covered population')
    ax2.set_ylabel('Covered population', color='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    formatter2 = ZeroAwareScalarFormatter(useOffset=False, useMathText=True)
    formatter2.set_scientific(True)
    formatter2.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(formatter2)

    # Assign colors to countries
    unique_countries = list(dict.fromkeys(countries_sorted))
    country_colors = {}
    for country in unique_countries:
        if country in COUNTRY_COLOR:
            country_colors[country] = COUNTRY_COLOR[country]
        else:
            cmap = plt.get_cmap('Dark2')
            country_colors[country] = cmap(len(country_colors) % cmap.N)

    # Set X-axis labels and color them by country
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities_sorted, rotation=90, ha='center')
    for lbl, ctry in zip(ax1.get_xticklabels(), countries_sorted):
        lbl.set_color(country_colors.get(ctry, '#333333'))

    # Main Legend
    color_area = '#4C72B0'
    color_pop = '#C44E52'

    legend_elements = [
        Patch(facecolor=color_area, edgecolor='k', label='Covered urban area (km²)'),
        Patch(facecolor=color_pop, edgecolor='k', label='Covered population')
    ]

    main_legend = ax1.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=1
    )
    ax1.add_artist(main_legend)

    # Total Coverage Box
    total_area = sum(covered_areas_sorted)
    total_population = sum(covered_populations_sorted)


    def format_number_str(num, is_pop=False):
        if num >= 1e6:
            return f"{num:,.0f}"
        elif num >= 1e3:
            return f"{num:,.1f}" if num % 1 != 0 else f"{int(num):,}"
        else:
            return f"{num:.1f}"


    area_str = f"{format_number_str(total_area)} km²"
    pop_str = f"{format_number_str(total_population)}"

    total_handles = [
        Patch(facecolor=color_area, edgecolor='k', label=area_str),
        Patch(facecolor=color_pop, edgecolor='k', label=pop_str)
    ]

    total_legend = ax1.legend(
        handles=total_handles,
        title="Total coverage",
        loc='upper left',
        bbox_to_anchor=(0.01, 0.98),
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=0.9
    )
    total_legend.get_title().set_ha("center")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "Area_Calc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Ratios / Percentages
    # ---------------------------------------------------------

    # Sort by area ratio
    sorted_pairs_ratio = sorted(zip(cities, area_ratios, pop_ratios, countries), key=lambda x: x[1])
    cities_r, area_ratios_r, pop_ratios_r, countries_r = zip(*sorted_pairs_ratio)

    x_r = np.arange(len(cities_r))

    fig2, ax3 = plt.subplots(figsize=(22, 8))

    # Add alternating background stripes
    for i in range(len(cities_r)):
        if i % 2 == 0:
            ax3.axvspan(i - 0.5, i + 0.5, facecolor='gray', alpha=0.3, zorder=0)

    # Left Y-axis: Area Ratio (blue)
    bars3 = ax3.bar(x_r - width / 2, area_ratios_r, width, color='#4C72B0', label='Covered urban area (%)')
    ax3.set_ylabel('Covered urban area (%)', color='#4C72B0')
    ax3.tick_params(axis='y', labelcolor='#4C72B0')
    ax3.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    # Right Y-axis: Population Ratio (red)
    ax4 = ax3.twinx()
    bars4 = ax4.bar(x_r + width / 2, pop_ratios_r, width, color='#C44E52', label='Covered population (%)')
    ax4.set_ylabel('Covered population (%)', color='#C44E52')
    ax4.tick_params(axis='y', labelcolor='#C44E52')
    ax4.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    # Set X-axis labels and color them by country
    ax3.set_xticks(x_r)
    ax3.set_xticklabels(cities_r, rotation=90, ha='center')
    for lbl, ctry in zip(ax3.get_xticklabels(), countries_r):
        lbl.set_color(country_colors.get(ctry, '#333333'))

    # Main Legend for Ratio Plot
    legend_elements_ratio = [
        Patch(facecolor=color_area, edgecolor='k', label='Covered urban area (%)'),
        Patch(facecolor=color_pop, edgecolor='k', label='Covered population (%)')
    ]

    main_legend_ratio = ax3.legend(
        handles=legend_elements_ratio,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=1
    )
    ax3.add_artist(main_legend_ratio)

    # Global Average Ratio Box
    # Calculate global totals to get a weighted average percentage
    global_covered_area = sum(df['intersect_area_m2'].fillna(0))
    global_total_area = sum(df['total_builtup_area_m2'].fillna(0))
    global_covered_pop = sum(df['estimated_population'].fillna(0))
    global_total_pop = sum(df['total_builtup_pop'].fillna(0))

    if global_total_area > 0:
        global_area_ratio = (global_covered_area / global_total_area) * 100
    else:
        global_area_ratio = 0

    if global_total_pop > 0:
        global_pop_ratio = (global_covered_pop / global_total_pop) * 100
    else:
        global_pop_ratio = 0

    area_ratio_str = f"{global_area_ratio:.1f}%"
    pop_ratio_str = f"{global_pop_ratio:.1f}%"

    total_handles_ratio = [
        Patch(facecolor=color_area, edgecolor='k', label=area_ratio_str),
        Patch(facecolor=color_pop, edgecolor='k', label=pop_ratio_str)
    ]

    total_legend_ratio = ax3.legend(
        handles=total_handles_ratio,
        title="Average coverage ratio",
        loc='upper left',
        bbox_to_anchor=(0.01, 0.98),
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=0.9
    )
    total_legend_ratio.get_title().set_ha("center")

    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "Area_Calc_Ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()