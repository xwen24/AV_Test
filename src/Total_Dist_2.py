import os
import shutil
import json
import warnings
warnings.filterwarnings('ignore')
import re
import folium
import geopandas as gpd
import scienceplots
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
from shapely.ops import unary_union
from utilities.normalize_longitude import normalize_longitude
from shapely.geometry import GeometryCollection
from utilities.hyperparameters import CHN_CITY_NAME


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


BUILT_UP_GPKG_PATH1 = "Data/QGIS/GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg"
LAYER1 = "GHS_UCDB_THEME_GHSL_GLOBE_R2024A"
ADMIN_SHP_PATH = 'Data/QGIS/ChinaAdminDivison/3. City/city.shp'


def load_built_up_data():
    if not os.path.exists(BUILT_UP_GPKG_PATH1):
        raise FileNotFoundError(f"Built-up GPKG not found at: {BUILT_UP_GPKG_PATH1}")
    print("Loading global built-up area data...")
    gdf = gpd.read_file(BUILT_UP_GPKG_PATH1, layer=LAYER1).to_crs("EPSG:4326")
    _ = gdf.sindex
    print("Global built-up data loaded.")
    return gdf


def ensure_valid_polygon(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if isinstance(geometry, GeometryCollection) or geometry.geom_type == 'GeometryCollection':
        polys = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        geometry = unary_union(polys)
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        if hasattr(geometry, 'geoms'):
            polys = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if polys:
                geometry = unary_union(polys)
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return None
    return geometry


def normalize_coords_shapely(geom):
    def norm_poly(p):
        ext = [(normalize_longitude(x), y) for x, y in p.exterior.coords]
        ints = [[(normalize_longitude(x), y) for x, y in r.coords] for r in p.interiors]
        return Polygon(ext, ints)

    if geom.geom_type == 'Polygon':
        return norm_poly(geom)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([norm_poly(p) for p in geom.geoms])
    return geom


def normalize_coords_json(coords):
    if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
        return [normalize_longitude(coords[0]), coords[1]]
    return [normalize_coords_json(c) for c in coords]


def merge_polygons(input_files, output_file):
    all_polygons = []
    for fp in input_files:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        feats = [data] if data['type'] == 'Feature' else data.get('features', [])
        for feat in feats:
            all_polygons.append(normalize_coords_shapely(shape(feat['geometry'])))
    result = {"type": "Feature", "properties": {}, "geometry": mapping(unary_union(all_polygons))}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def extract_city_corp(filename):
    parts = filename.split('-')
    if parts[0] == 'may' and parts[1] == 'mobility':
        return ' '.join(parts[2:-4]).title(), 'May mobility'
    elif parts[0] in ['avride', 'moia', 'tesla', 'waymo', 'zoox']:
        return ' '.join(parts[1:-4]).title(), parts[0]
    elif parts[1:3] == ['abu', 'dhabi']:
        return ' '.join(parts[1:-4]).title(), 'WeRide'
    return ' '.join(parts[:-1]).title(), 'Unknown'


def save_static_map(gdf, out_path, title, color, alpha=0.6):
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.boundary.plot(ax=ax, color='black', zorder=2)
    gdf.plot(ax=ax, color=color, alpha=alpha, edgecolor='black', zorder=1)

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    try:
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Warning: Basemap failed: {e}")
        ax.set_box_aspect(1)
    ax.set(xlabel='Longitude', ylabel='Latitude', title=title)
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static map saved: {out_path}")


def save_interactive_map(geojson_data, center, name, out_path):
    m = folium.Map(location=[center.y, center.x], zoom_start=11)
    folium.GeoJson(
        geojson_data, name=name,
        style_function=lambda x: {'fillColor': 'blue', 'color': 'darkblue', 'weight': 2, 'fillOpacity': 0.4},
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Area: ']) if 'name' in geojson_data.get('properties',
                                                                                                         {}) else None
    ).add_to(m)
    m.save(out_path)
    print(f"Interactive map saved: {out_path}")


def standardize_city_name(name: str) -> str:
    alias_map = {"bay area": "San Francisco", "silicon valley": "San Francisco"}
    name_clean = name.strip()
    name_lower = name_clean.lower()
    for alias, canonical in alias_map.items():
        if alias in name_lower:
            name_clean = canonical
            break
    return re.sub(r'\s*\d+$', '', name_clean).strip()


# --- Main Execution ---
built_up_gdf1 = load_built_up_data()

# Load Administrative Boundaries if needed (only once)
admin_gdf = None
if any(CHN_CITY_NAME.values()):
    if os.path.exists(ADMIN_SHP_PATH):
        print(f"Loading administrative boundaries from {ADMIN_SHP_PATH}...")
        admin_gdf = gpd.read_file(ADMIN_SHP_PATH)
        if admin_gdf.crs != "EPSG:4326":
            print("Reprojecting administrative boundaries to EPSG:4326...")
            admin_gdf = admin_gdf.to_crs("EPSG:4326")
        print(f"Loaded administrative boundaries for {len(admin_gdf)} features.")
    else:
        print(f"Warning: Administrative boundary shapefile not found at {ADMIN_SHP_PATH}. Skipping admin clipping.")

# 1. Organize raw files
city_lists = {
    "GER": ["moia-hamburg-july-1-2025-boundary.geojson"],
    "UAE": ["weride-abu-dhabi-july-29-2025-boundary.geojson"],
    "USA": [
        "avride-dallas-december-3-2025-boundary.geojson",
        "may-mobility-arlington-tx-march-23-2021-boundary.geojson",
        "may-mobility-atlanta-september-10-2025-boundary.geojson",
        "tesla-austin-october-28-2025-boundary.geojson",
        "tesla-bay-area-july-31-2025-boundary.geojson",
        "waymo-atlanta-june-24-2025-boundary.geojson",
        "waymo-austin-july-17-2025-boundary.geojson",
        "waymo-los-angeles-june-18-2025-boundary.geojson",
        "waymo-phoenix-june-5-2024-boundary.geojson",
        "waymo-san-francisco-june-17-2025-boundary.geojson",
        "waymo-silicon-valley-november-12-2025-boundary.geojson",
        "zoox-san-francisco-november-18-2025-boundary.geojson"
    ]
}

for country, files in city_lists.items():
    target_dir = f"Data/raw/{country}/GeoJSON/"
    os.makedirs(target_dir, exist_ok=True)
    for fname in files:
        src = os.path.join("Data/raw/USA/geometries/", fname)
        if os.path.exists(src):
            city, corp = extract_city_corp(fname)
            shutil.copy2(src, os.path.join(target_dir, f"{city}-{corp.title()}.geojson"))
            print(f"Copied: {fname}")
        else:
            print(f"Missing: {fname}")

# 2. Set up output dirs
RESULTS_BASE = "src/Results/Total_Dist"
for d in ["Corp", "City", "Built_Up", "Built_Up_Intersect"]:
    os.makedirs(os.path.join(RESULTS_BASE, d), exist_ok=True)

# 3. Process Corp-level (by country)
for country in ["CHN", "GER", "SK", "UAE", "USA"]:
    src_path = f"Data/raw/{country}/GeoJSON/"
    if not os.path.exists(src_path):
        continue
    for item in os.listdir(src_path):
        if not item.endswith(".geojson"):
            continue
        print(f"Processing Corp: {country}/{item}")
        with open(os.path.join(src_path, item), 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "geometry" not in data:
            feat = data["features"][0]
            data = {"type": "Feature", "properties": {"name": item[:-8]}, "geometry": feat["geometry"]}
        data["geometry"]["coordinates"] = normalize_coords_json(data["geometry"]["coordinates"])

        gdf = gpd.GeoDataFrame.from_features([data], crs="EPSG:4326")
        name = item[:-8]

        save_interactive_map(data, gdf.geometry.unary_union.centroid, name,
                             os.path.join(RESULTS_BASE, "Corp", f"{name}.html"))
        save_static_map(gdf, os.path.join(RESULTS_BASE, "Corp", f"{name}.png"),
                        f"{name} Service Area", '#1f77b4')

# 4. Process City-level (once, globally)
corp_dir = os.path.join(RESULTS_BASE, "Corp")
raw_cities = {f.split("-")[0] for f in os.listdir(corp_dir) if f.endswith(".png")}
std_cities = {standardize_city_name(c) for c in raw_cities}
print(f"\nMerging cities: {sorted(std_cities)}")

city_to_files = {city: [] for city in std_cities}
for country in ["CHN", "GER", "SK", "UAE", "USA"]:
    raw_dir = f"Data/raw/{country}/GeoJSON/"
    if not os.path.exists(raw_dir):
        continue
    for f in os.listdir(raw_dir):
        if not f.endswith(".geojson"):
            continue
        raw_city = f.split("-")[0]
        std_city = standardize_city_name(raw_city)
        if std_city in city_to_files:
            city_to_files[std_city].append(os.path.join(raw_dir, f))

for city, files in city_to_files.items():
    if files:
        print(f"Merging {len(files)} files for: {city}")
        merge_polygons(files, os.path.join(RESULTS_BASE, "City", f"{city}.geojson"))

# 5. Process merged cities + built-up
city_path = os.path.join(RESULTS_BASE, "City")
intersect_raw_path = os.path.join(RESULTS_BASE, "Built_Up_Intersect")
built_up_base_path = os.path.join(RESULTS_BASE, "Built_Up")
no_overlap_cities = []

for item in sorted(f for f in os.listdir(city_path) if f.endswith(".geojson")):
    print(f"Processing City: {item}")
    with open(os.path.join(city_path, item), 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "geometry" not in data:
        feat = data["features"][0]
        data = {"type": "Feature", "properties": {"name": item[:-8]}, "geometry": feat["geometry"]}
    data["geometry"]["coordinates"] = normalize_coords_json(data["geometry"]["coordinates"])

    gdf = gpd.GeoDataFrame.from_features([data], crs="EPSG:4326")
    name = item[:-8]

    save_interactive_map(data, gdf.geometry.unary_union.centroid, name,
                         os.path.join(city_path, f"{name}.html"))
    save_static_map(gdf, os.path.join(city_path, f"{name}.png"),
                    f"{name} Service Area", '#1f77b4')

    # --- NEW: Built_Up_Intersect logic (raw intersecting built-up polygons, with admin clipping for CHN) ---
    minx, miny, maxx, maxy = gdf.total_bounds
    bu_subset1 = built_up_gdf1.cx[minx:maxx, miny:maxy]

    final_intersect_gdf = gpd.GeoDataFrame()

    if not bu_subset1.empty:
        # Get all built-up polygons that intersect the service area (raw, unclipped)
        raw_intersect_gdf = gpd.sjoin(bu_subset1, gdf, how="inner", predicate="intersects")
        raw_intersect_gdf = raw_intersect_gdf[built_up_gdf1.columns].drop_duplicates().reset_index(drop=True)

        if not raw_intersect_gdf.empty:
            if name in CHN_CITY_NAME and admin_gdf is not None:
                admin_name_to_find = CHN_CITY_NAME[name]
                print(
                    f"  > Found {name} in CHN_CITY_NAME. Clipping raw built-up polygons with admin boundary: '{admin_name_to_find}'")

                admin_polygons = admin_gdf[admin_gdf['ct_name'] == admin_name_to_find]
                if not admin_polygons.empty:
                    admin_boundary = unary_union(admin_polygons.geometry)
                    admin_boundary = ensure_valid_polygon(admin_boundary)

                    if admin_boundary is not None:
                        clipped_geoms = []
                        for geom in raw_intersect_gdf.geometry:
                            if geom.is_valid and not geom.is_empty:
                                clipped = geom.intersection(admin_boundary)
                                clipped = ensure_valid_polygon(clipped)
                                if clipped is not None and not clipped.is_empty:
                                    clipped_geoms.append(clipped)

                        if clipped_geoms:
                            final_intersect_gdf = gpd.GeoDataFrame(
                                raw_intersect_gdf.drop(columns='geometry').iloc[:len(clipped_geoms)].reset_index(
                                    drop=True),
                                geometry=clipped_geoms,
                                crs="EPSG:4326"
                            )
                            print(f"  > Admin-clipped {len(clipped_geoms)} built-up polygons for {name}.")
                        else:
                            print(f"  > Warning: No valid geometries after admin clipping for {name}.")
                    else:
                        print(f"  > Invalid admin boundary for {name}. Using raw intersect polygons.")
                        final_intersect_gdf = raw_intersect_gdf.copy()
                else:
                    print(f"  > Admin boundary not found for '{admin_name_to_find}'. Using raw intersect polygons.")
                    final_intersect_gdf = raw_intersect_gdf.copy()
            else:
                final_intersect_gdf = raw_intersect_gdf.copy()
    else:
        print(f"  > No built-up polygons intersect with service area for {name}.")

    # Save Built_Up_Intersect results
    out_intersect_geojson = os.path.join(intersect_raw_path, f"{name}.geojson")
    out_intersect_png = os.path.join(intersect_raw_path, f"{name}.png")

    if not final_intersect_gdf.empty:
        final_intersect_gdf.to_file(out_intersect_geojson, driver="GeoJSON")
        save_static_map(
            final_intersect_gdf,
            out_intersect_png,
            f"Built-up Polygons Intersecting {name} (Raw{' + Admin-Clipped' if name in CHN_CITY_NAME else ''})",
            '#ff7f0e',
            alpha=0.5
        )
        print(f"Saved Built_Up_Intersect for {name} to {out_intersect_geojson}")
    else:
        # Save empty GeoJSON
        empty_feature = {
            "type": "Feature",
            "properties": {"city": name},
            "geometry": None
        }
        with open(out_intersect_geojson, 'w', encoding='utf-8') as f:
            json.dump(empty_feature, f, ensure_ascii=False, indent=2)
        print(f"No built-up polygons intersect (after admin clipping) for {name}. Empty GeoJSON saved.")

    # --- ORIGINAL Built_Up logic (UNCHANGED) ---
    intersect = gpd.overlay(gdf, bu_subset1, how='intersection') if not bu_subset1.empty else gpd.GeoDataFrame()
    if not intersect.empty:
        out_base_initial = os.path.join(built_up_base_path, name)
        merged_geom = unary_union(intersect.geometry)
        merged_geom = normalize_coords_shapely(merged_geom)
        merged_geom = ensure_valid_polygon(merged_geom)

        if merged_geom is not None:
            result_feature = {
                "type": "Feature",
                "properties": {"city": name},
                "geometry": mapping(merged_geom)
            }
            with open(f"{out_base_initial}.geojson", 'w', encoding='utf-8') as f:
                json.dump(result_feature, f, ensure_ascii=False, indent=2)

            intersect_gdf = gpd.GeoDataFrame([1], geometry=[merged_geom], crs="EPSG:4326")
            save_static_map(intersect_gdf, f"{out_base_initial}.png", f"Built-up Area in {name}", '#2ca02c', alpha=0.7)
        else:
            print(f"  > Error: Initial built-up intersection for {name} is invalid after cleaning.")
    else:
        print(f"No built-up overlap for city: {name} (overlay result)")
        no_overlap_cities.append(item)

# --- Save Readme for No Intersection ---
readme_path = os.path.join(RESULTS_BASE, "readme.txt")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("Files with no intersection with Built-Up Data:\n")
    f.write("==============================================\n")
    if no_overlap_cities:
        for city_file in no_overlap_cities:
            f.write(f"{city_file}\n")
    else:
        f.write("None. All cities intersect with built-up data.\n")
print(f"Summary of non-intersecting files saved to: {readme_path}")