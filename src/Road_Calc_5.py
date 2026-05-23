import os
from pathlib import Path
import gc
import geopandas as gpd
import osmnx as ox
import networkx as nx
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import contextily as ctx
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from utilities.hyperparameters import ROAD_TYPES, COUNTRY_COLOR, CHN_CITY_NAME
import concurrent.futures
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
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


def ensure_valid_polygon(geometry):
    """
    Helper function: Ensures the geometry is a valid Polygon or MultiPolygon.
    Handles GeometryCollections and invalid topologies.
    """
    if geometry is None or geometry.is_empty:
        return None

    # 1. Attempt to fix validity (e.g., self-intersections)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    # 2. If it's a GeometryCollection (mixed types), extract only Polygons/MultiPolygons
    if isinstance(geometry, GeometryCollection) or geometry.geom_type == 'GeometryCollection':
        polys = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        geometry = unary_union(polys)

    # 3. Final check: must be Polygon or MultiPolygon
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        # Try to extract again if buffer(0) changed the type unexpectedly
        if hasattr(geometry, 'geoms'):
            polys = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if polys:
                geometry = unary_union(polys)

    # 4. If still not valid, return None or try buffer(0) one last time
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return None

    return geometry


def get_and_save_graph(unified_polygon, city_name, output_dir, road_types):
    """
    Downloads the road network, saves it locally, and returns the Graph object.
    """
    unified_polygon = ensure_valid_polygon(unified_polygon)

    if unified_polygon is None:
        print(f"Error: Valid Polygon/MultiPolygon could not be created for {city_name}")
        return None

    # 1. Build Filter (including _link)
    final_road_types = set(road_types)
    for road_type in road_types:
        if not road_type.endswith('_link'):
            final_road_types.add(f"{road_type}_link")
    filter_str = "|".join(sorted(list(final_road_types)))
    custom_filter = f'["highway"~"{filter_str}"]'

    # 2. Prepare save path
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{city_name}.graphml")

    print(f"Downloading graph for {city_name} with filter: {custom_filter}")
    try:
        G = ox.graph_from_polygon(unified_polygon,
                                  custom_filter=custom_filter,
                                  simplify=True,
                                  truncate_by_edge=True)
        ox.save_graphml(G, save_path)
        print(f"Graph saved to: {save_path}")
        return G
    except Exception as e:
        print(f"Error downloading/saving graph for {city_name}: {e}")
        return None


def process_single_polygon(row, idx, total_polys, output_dir, road_types):
    """
    辅助函数：处理单个多边形的逻辑，用于多线程调用。
    """
    try:
        col1_name = 'ID_UC_G0'
        col2_name = 'GC_UCN_MAI_2025'
        col1_val = row.get(col1_name)
        col2_val = row.get(col2_name)

        poly_name = f"{col1_val}_{col2_val}"
        print(f"[{idx + 1}/{total_polys}] Starting: {poly_name}")

        # Clean geometry
        polygon = ensure_valid_polygon(row.geometry)
        if polygon is None:
            return f"Skipped (Invalid Geometry): {poly_name}"

        out_path = os.path.join(output_dir, f"{poly_name}.graphml")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{idx + 1}/{total_polys}] Skip (exists): {poly_name}.graphml")
            return f"Skip (exists): {poly_name}"

        get_and_save_graph(polygon, poly_name, output_dir, road_types)

        return f"Success: {poly_name}"

    except Exception as e:
        error_msg = f"Error processing polygon {idx}: {e}"
        print(error_msg)
        return error_msg


def process_gpkg_polygons(gpkg_path, output_dir, road_types, max_workers=4):
    """
    读取 GPKG 文件中的多个多边形，使用多线程并行下载并保存路网数据。
    """
    if not os.path.exists(gpkg_path):
        print(f"GPKG file not found: {gpkg_path}")
        return

    print(f"\n--- Processing GPKG: {gpkg_path} with {max_workers} workers ---")
    try:
        gdf = gpd.read_file(gpkg_path)
        gdf.columns = gdf.columns.str.replace(r'^\ufeff', '', regex=True)

        if gdf.crs != "EPSG:4326":
            print("Reprojecting GPKG to EPSG:4326...")
            gdf = gdf.to_crs("EPSG:4326")

        total_polys = len(gdf)
        print(f"Found {total_polys} polygons in GPKG.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, row in gdf.iterrows():
                future = executor.submit(
                    process_single_polygon,
                    row,
                    idx,
                    total_polys,
                    output_dir,
                    road_types
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        print("\n--- All GPKG polygons processed ---")

    except Exception as e:
        print(f"Error processing GPKG file: {e}")


def count_roundabouts(G):
    if not G or G.number_of_edges() == 0:
        return 0

    roundabout_edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get('junction') == 'roundabout':
            roundabout_edges.append((u, v, k))

    if not roundabout_edges:
        return 0

    roundabout_subgraph = G.edge_subgraph(roundabout_edges).to_undirected()
    return nx.number_connected_components(roundabout_subgraph)


def calculate_inter_number(G, road_types=ROAD_TYPES, min_street_count=3):
    if not G or not getattr(G, 'nodes', None) or G.number_of_nodes() == 0:
        return 0, 0, 0, 0, 0

    try:
        num_roundabouts = count_roundabouts(G)

        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        potential_nodes = nodes_gdf[nodes_gdf['street_count'] >= min_street_count].copy()

        if potential_nodes.empty:
            num_inter = 0
        else:
            node_has_target_road = set()
            for u, v, data in G.edges(data=True):
                hwy = data.get('highway', 'unclassified')
                hwy_set = {hwy} if isinstance(hwy, str) else set(hwy)
                if any(rt in hwy_set for rt in road_types):
                    node_has_target_road.add(u)
                    node_has_target_road.add(v)

            roundabout_nodes = set()
            roundabout_edges = []
            for u, v, k, data in G.edges(keys=True, data=True):
                if data.get('junction') == 'roundabout':
                    roundabout_edges.append((u, v, k))
            if roundabout_edges:
                roundabout_subgraph = G.edge_subgraph(roundabout_edges)
                roundabout_nodes = set(roundabout_subgraph.nodes())

            valid_regular = potential_nodes[
                (potential_nodes.index.isin(node_has_target_road)) &
                (~potential_nodes.index.isin(roundabout_nodes))
                ]
            num_inter = len(valid_regular)
            num_multi_arm = len(valid_regular[valid_regular['street_count'] > 4])

            traffic_signals = 0
            stop_signs = 0
            for node_id in valid_regular.index:
                node_attrs = G.nodes[node_id]
                hw = node_attrs.get('highway')
                if hw == 'traffic_signals':
                    traffic_signals += 1
                elif hw == 'stop':
                    stop_signs += 1

        print(
            f"Intersections: {num_inter}, Roundabouts: {num_roundabouts}, Signals: {traffic_signals}, Stops: {stop_signs}")
        return num_inter, traffic_signals, stop_signs, num_multi_arm, num_roundabouts

    except Exception as e:
        print(f"Error in count_num_intersections_and_controls: {e}")
        return 0, 0, 0, 0, 0


def calculate_road_length(G, unified_polygon, city_name, road_types, include_related_links,
                          num_inter, output_dir, traffic_counts):
    """
    计算路网物理长度（用于道路密度分析）。
    """
    if not G or len(G.edges) == 0:
        return {rt: 0.0 for rt in road_types} | {'total': 0.0}

    # Ensure geometry is clean for clipping
    unified_polygon = ensure_valid_polygon(unified_polygon)
    if unified_polygon is None:
        print(f"Error: Invalid geometry for clipping in {city_name}")
        return {rt: 0.0 for rt in road_types} | {'total': 0.0}

    G_undir = G.to_undirected()
    edges_gdf = ox.graph_to_gdfs(G_undir, nodes=False, edges=True)

    # 2. Clip (Ensure we use the clean MultiPolygon/Polygon)
    try:
        clipped_edges_gdf = gpd.clip(edges_gdf, unified_polygon)
    except Exception as e:
        print(f"Error clipping edges for {city_name}: {e}")
        return {rt: 0.0 for rt in road_types} | {'total': 0.0}

    if clipped_edges_gdf.empty:
        print("Warning: No roads found after clipping.")
        return {rt: 0.0 for rt in road_types} | {'total': 0.0}

    # 3. Projection and Length
    utm_crs = clipped_edges_gdf.estimate_utm_crs()
    clipped_edges_proj = clipped_edges_gdf.to_crs(utm_crs)
    clipped_edges_proj['length'] = clipped_edges_proj.geometry.length

    if 'highway' in clipped_edges_proj.columns:
        if clipped_edges_proj['highway'].apply(lambda x: isinstance(x, list)).any():
            clipped_edges_proj = clipped_edges_proj.explode('highway')

    lengths_by_type_m = clipped_edges_proj.groupby('highway', dropna=True)['length'].sum()
    lengths_by_type_km = (lengths_by_type_m / 1000).to_dict()

    road_length_by_type = {road_type: 0.0 for road_type in road_types}
    for road_type in road_types:
        length = lengths_by_type_km.get(road_type, 0.0)
        if include_related_links and not road_type.endswith('_link'):
            length += lengths_by_type_km.get(f"{road_type}_link", 0.0)
        road_length_by_type[road_type] = length

    total_length_km = sum(road_length_by_type.values())
    road_length_by_type['total'] = total_length_km

    # 4. Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_data = [{'road_type': rt, 'value': l} for rt, l in road_length_by_type.items() if rt != 'total']
    csv_data.extend([
        {'road_type': 'total', 'value': road_length_by_type['total']},
        {'road_type': 'num_inter', 'value': num_inter},
        {'road_type': 'roundabouts', 'value': traffic_counts['roundabouts']},
        {'road_type': 'traffic_signals', 'value': traffic_counts['traffic_signals']},
        {'road_type': 'stop_signs', 'value': traffic_counts['stop_signs']}
    ])
    df = pd.DataFrame(csv_data)
    csv_filename = os.path.join(output_dir, f"{city_name}_road_mileage.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Road mileage data saved to: {csv_filename}")

    # 6. Visualization
    if not clipped_edges_proj.empty:
        print("Generating visualizations...")
        fig_map, ax_map = None, None
        fig_bar, ax_bar = None, None

        try:
            fig_map, ax_map = plt.subplots(figsize=(15, 10))
            clipped_edges_wgs84 = clipped_edges_proj.to_crs(epsg=4326)

            if not clipped_edges_wgs84.empty:
                # Calculate bounds based on ALL edges (which cover all parts of a MultiPolygon)
                minx, miny, maxx, maxy = clipped_edges_wgs84.total_bounds
                margin_x = (maxx - minx) * 0.05
                margin_y = (maxy - miny) * 0.05
                ax_map.set_xlim(minx - margin_x, maxx + margin_x)
                ax_map.set_ylim(miny - margin_y, maxy + margin_y)
                ax_map.set_box_aspect(1)
            else:
                print("Warning: No edges to plot, skipping map generation.")

            unique_main_types = sorted(road_types)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_main_types)))
            color_map = {rt: color for rt, color in zip(unique_main_types, colors)}

            if clipped_edges_wgs84['highway'].apply(lambda x: isinstance(x, list)).any():
                plot_edges = clipped_edges_wgs84.explode('highway')
            else:
                plot_edges = clipped_edges_wgs84

            for road_type, group_data in plot_edges.groupby('highway'):
                main_type = str(road_type).replace('_link', '')
                color = color_map.get(main_type, 'gray')
                # Plotting handles disconnected parts automatically if they are in the dataframe
                group_data.plot(ax=ax_map, linewidth=1.5, edgecolor=color, label=main_type)

            ctx.add_basemap(ax_map, crs=clipped_edges_wgs84.crs, source=ctx.providers.CartoDB.Positron)

            from matplotlib.lines import Line2D
            legend_elements = []
            for rt in ROAD_TYPES:
                if road_length_by_type.get(rt, 0) > 0:
                    main_type = rt.replace('_link', '')
                    color = color_map.get(main_type, 'gray')
                    legend_elements.append(
                        Line2D([0], [0], color=color, lw=2, label=f'{rt} ({road_length_by_type[rt]:.2f} km)')
                    )

            ax_map.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Road types")
            ax_map.set_xlabel('Longitude')
            ax_map.set_ylabel('Latitude')
            plt.tight_layout()

            map_filename = os.path.join(output_dir, f"{city_name}_road_map.png")
            fig_map.savefig(map_filename, dpi=300, bbox_inches='tight')
            print(f"Map saved as: {map_filename}")

            # Bar chart
            plot_data_ordered = {}
            for rt in ROAD_TYPES:
                if rt in road_length_by_type and road_length_by_type[rt] > 0:
                    plot_data_ordered[rt] = road_length_by_type[rt]

            if not plot_data_ordered:
                print("No data to plot for bar chart.")
            else:
                fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
                bars = ax_bar.bar(plot_data_ordered.keys(), plot_data_ordered.values())

                ax_bar.set_xticklabels(plot_data_ordered.keys(), rotation=45, ha='right')
                ax_bar.set_xlabel('Road types')
                ax_bar.set_ylabel('Mileage (km)')

                for bar in bars:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f'{bar.get_height():.1f}', ha='center', va='bottom')

                plt.tight_layout()
                bar_filename = os.path.join(output_dir, f"{city_name}_road_type.png")
                fig_bar.savefig(bar_filename, dpi=300, bbox_inches='tight')
                print(f"Bar chart saved as: {bar_filename}")
        finally:
            if fig_map: plt.close(fig_map)
            if fig_bar: plt.close(fig_bar)

    return road_length_by_type


def process_uncovered_area(deployed_polygon, city_name, gpkg_gdf, admin_gdf, graphml_uncovered, csv_uncovered,
                           road_types):
    print(f"Processing Uncovered Area for {city_name}...")

    deployed_polygon = ensure_valid_polygon(deployed_polygon)
    if deployed_polygon is None:
        print(f"Error: Deployed polygon for {city_name} is invalid.")
        return

    # 1. Geometry Calculation
    possible_matches_index = list(gpkg_gdf.sindex.query(deployed_polygon, predicate='intersects'))
    relevant_gpkg_polys = gpkg_gdf.iloc[possible_matches_index]

    if relevant_gpkg_polys.empty:
        print(f"Warning: No intersecting GPKG polygons found for {city_name}.")
        return

    # Merge full city boundary and clean it
    full_city_polygon = unary_union(relevant_gpkg_polys.geometry)
    full_city_polygon = ensure_valid_polygon(full_city_polygon)

    if full_city_polygon is None:
        print(f"Error: Could not create valid full city polygon for {city_name}")
        return

    # Calculate Uncovered Area
    try:
        uncovered_polygon = full_city_polygon.difference(deployed_polygon)
        uncovered_polygon = ensure_valid_polygon(uncovered_polygon)  # Clean result
    except Exception as e:
        print(f"Error calculating difference for {city_name}: {e}")
        return

    if uncovered_polygon is None or uncovered_polygon.is_empty:
        print(f"Info: {city_name} is fully covered by the deployed area.")
        return

    # 2. Conditional Administrative Boundary Clipping
    admin_boundary = None

    if city_name in CHN_CITY_NAME:
        admin_name_to_find = CHN_CITY_NAME[city_name]
        print(f"  > Found {city_name} in CHN_CITY_NAME list. Using administrative name: '{admin_name_to_find}'")

        admin_polygons = admin_gdf[admin_gdf['ct_name'] == admin_name_to_find]

        if admin_polygons.empty:
            print(
                f"Warning: No administrative boundary found for '{admin_name_to_find}'. Proceeding with original uncovered area.")
            final_area_polygon = uncovered_polygon
        else:
            admin_boundary = unary_union(admin_polygons.geometry)
            admin_boundary = ensure_valid_polygon(admin_boundary)  # Clean admin boundary

            print(f"  > Clipping uncovered area with administrative boundary...")
            try:
                final_area_polygon = uncovered_polygon.intersection(admin_boundary)
                final_area_polygon = ensure_valid_polygon(final_area_polygon)  # Clean result
            except Exception as e:
                print(f"  > Error intersecting uncovered area with administrative boundary: {e}")
                return

            if final_area_polygon is None or final_area_polygon.is_empty:
                print(f"  > Info: The uncovered area within the administrative boundary is empty.")
                return
    else:
        print(f"  > {city_name} not in CHN_CITY_NAME list. Skipping administrative boundary clipping.")
        final_area_polygon = uncovered_polygon

    # 3. Check if GraphML file already exists
    os.makedirs(graphml_uncovered, exist_ok=True)
    save_path = os.path.join(graphml_uncovered, f"{city_name}.graphml")

    if os.path.exists(save_path):
        print(f"  > GraphML file already exists: {save_path}. Skipping download and proceeding to step 6.")
        # Load the existing graph
        try:
            G_final_area = ox.load_graphml(save_path)
        except Exception as e:
            print(f"  > Error loading existing graph: {e}")
            return
    else:
        # 3. Download Full City Graph
        print(f"  > Downloading WHOLE city graph into memory...")

        final_road_types = set(road_types)
        for road_type in road_types:
            if not road_type.endswith('_link'):
                final_road_types.add(f"{road_type}_link")
        filter_str = "|".join(sorted(list(final_road_types)))
        custom_filter = f'["highway"~"{filter_str}"]'

        G_total = None
        try:
            # Download using the CLEAN full city polygon
            G_total = ox.graph_from_polygon(full_city_polygon,
                                            custom_filter=custom_filter,
                                            simplify=True,
                                            truncate_by_edge=True)
        except Exception as e:
            print(f"  > Error downloading whole city graph: {e}")
            return

        if G_total is None or len(G_total.nodes) == 0:
            print("  > Downloaded graph is empty.")
            return

        # 4. Clip Graph to Final Area
        print(f"  > Clipping graph to the final area...")
        G_final_area = None
        try:
            # Truncate using the CLEAN final area polygon
            G_final_area = ox.truncate.truncate_graph_polygon(G_total, final_area_polygon)
        except Exception as e:
            print(f"  > Error clipping graph: {e}")
            del G_total
            gc.collect()
            return

        print("  > Cleaning up whole city graph from memory...")
        del G_total
        gc.collect()

        if len(G_final_area.nodes) == 0:
            print("  > Warning: Final area graph is empty after clipping.")
            return

        ox.save_graphml(G_final_area, save_path)
        print(f"  > Final area graph saved to: {save_path}")

    # 6. Calculate Metrics and Save CSV
    print(f"  > Calculating statistics for the final area...")

    try:
        num_inter, traffic_signals, stop_signs, num_multi_arm, num_roundabouts = calculate_inter_number(
            G_final_area, road_types, min_street_count=3
        )
        traffic_counts = {
            'traffic_signals': traffic_signals,
            'stop_signs': stop_signs,
            'roundabouts': num_roundabouts
        }

        calculate_road_length(
            G=G_final_area,
            unified_polygon=final_area_polygon,  # Pass the CLEAN polygon
            city_name=city_name,
            road_types=road_types,
            include_related_links=True,
            num_inter=num_inter,
            output_dir=csv_uncovered,
            traffic_counts=traffic_counts
        )
    except Exception as e:
        print(f"Error during calculation: {e}")
    finally:
        print("  > Final memory cleanup...")
        if 'G_final_area' in locals(): del G_final_area
        if 'final_area_polygon' in locals(): del final_area_polygon
        if 'uncovered_polygon' in locals(): del uncovered_polygon
        if 'full_city_polygon' in locals(): del full_city_polygon
        if 'admin_boundary' in locals(): del admin_boundary
        gc.collect()


def safe_get_value(df, key, default=0):
    try:
        return float(df.loc[key, 'value']) if key in df.index else default
    except (KeyError, ValueError, TypeError):
        return default


if __name__ == '__main__':
    plot_Road_Calc = True

    source_dir = 'src/Results/Total_Dist/Built_Up/'
    output_dir = 'src/Results/Road_Calc/AV_Accessible'
    osm_dir = 'Data/OSM/AV_Accessible'
    graphml_uncovered = 'Data/OSM/AV_Inaccessible_Uncovered'
    csv_uncovered = 'src/Results/Road_Calc/AV_Inaccessible_Uncovered'
    admin_shp_path = 'Data/QGIS/ChinaAdminDivison/3. City/city.shp'

    if not plot_Road_Calc:
        gpkg_path = 'Data/QGIS/GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg'
        gpkg_gdf = gpd.read_file(gpkg_path)
        if gpkg_gdf.crs != "EPSG:4326":
            gpkg_gdf = gpkg_gdf.to_crs("EPSG:4326")

        print(f"Reading administrative boundaries from {admin_shp_path}...")
        admin_gdf = gpd.read_file(admin_shp_path)
        if admin_gdf.crs != "EPSG:4326":
            print("Reprojecting administrative boundaries to EPSG:4326...")
            admin_gdf = admin_gdf.to_crs("EPSG:4326")
        print(f"Loaded administrative boundaries for {len(admin_gdf)} features.")

        if not os.path.isdir(source_dir):
            print(f"Error: Source directory not found at '{source_dir}'")
        else:
            for filename in os.listdir(source_dir):
                if filename.endswith('.geojson'):
                    geojson_file_path = os.path.join(source_dir, filename)
                    city_name = os.path.splitext(filename)[0]

                    # 将 Xiongan 改为 Xiong'an
                    if city_name == 'Xiongan':
                        city_name = "Xiong'an"

                    print(f"\n--- Processing {city_name} ---")

                    try:
                        gdf_polygons = gpd.read_file(geojson_file_path).to_crs("EPSG:4326")
                        raw_polygon = gdf_polygons.geometry.unary_union
                        unified_polygon = ensure_valid_polygon(raw_polygon)

                        if unified_polygon is None or unified_polygon.is_empty:
                            print(f"Skipping {city_name}: Invalid geometry after cleaning.")
                            continue

                    except Exception as e:
                        print(f"Error reading GeoJSON: {e}")
                        continue

                    # Download and save Graph (using clean polygon) - with check for existing file
                    graph_path = os.path.join(osm_dir, f"{city_name}.graphml")

                    # Check if GraphML file already exists
                    if os.path.exists(graph_path):
                        print(f"GraphML file already exists for {city_name}, loading from file...")
                        try:
                            G = ox.load_graphml(graph_path)
                            print(f"Successfully loaded existing graph for {city_name}")
                        except Exception as e:
                            print(f"Error loading existing graph for {city_name}: {e}")
                            print("Downloading new graph...")
                            G = get_and_save_graph(unified_polygon, city_name, osm_dir, ROAD_TYPES)
                    else:
                        # File doesn't exist, download and save
                        G = get_and_save_graph(unified_polygon, city_name, osm_dir, ROAD_TYPES)

                    if not G:
                        continue

                    num_inter, traffic_signals, stop_signs, num_multi_arm, num_roundabouts = calculate_inter_number(
                        G, ROAD_TYPES, min_street_count=3
                    )
                    traffic_counts = {'traffic_signals': traffic_signals, 'stop_signs': stop_signs,
                                      'roundabouts': num_roundabouts}

                    # Calculate mileage and plot (using clean polygon)
                    road_lengths = calculate_road_length(
                        G, unified_polygon, city_name,
                        road_types=ROAD_TYPES,
                        include_related_links=True,
                        num_inter=num_inter,
                        output_dir=output_dir,
                        traffic_counts=traffic_counts
                    )

                    if road_lengths:
                        print("\n--- Summary for", city_name, "---")
                        total = road_lengths.pop('total', 0)
                        sorted_types = dict(sorted(road_lengths.items(), key=lambda x: x[1], reverse=True))
                        for road_type, length in sorted_types.items():
                            print(f"{road_type:<15}: {length:.4f} km")
                        print("-" * 30)
                        print(f"{'Total':<15}: {total:.4f} km")
                        print("--------------------------------\n")

                    # Process Uncovered Area (using clean polygon)
                    print("--> Processing Uncovered Area")
                    process_uncovered_area(unified_polygon, city_name, gpkg_gdf, admin_gdf, graphml_uncovered,
                                           csv_uncovered, ROAD_TYPES)

    else:
        cache_path = "city_country_cache.json"
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                CITY_COUNTRY_CACHE = json.load(f)
        else:
            CITY_COUNTRY_CACHE = {}


        def normalize_city(name: str) -> str:
            return name.strip().lower()


        geolocator = Nominatim(user_agent="your-app-name-city-country", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)


        def infer_country_from_city(city_name: str) -> str:
            key = normalize_city(city_name)
            if key in CITY_COUNTRY_CACHE:
                return CITY_COUNTRY_CACHE[key]
            loc = geocode(city_name, addressdetails=True)
            country = "Unknown"
            if loc and hasattr(loc, "raw"):
                addr = loc.raw.get("address", {})
                if "country" in addr:
                    country = addr["country"]
                elif "country_code" in addr:
                    country = addr["country_code"].upper()
            CITY_COUNTRY_CACHE[key] = country
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(CITY_COUNTRY_CACHE, f, ensure_ascii=False, indent=2)
            return country


        cities = ["Xiong'an"]
        road_lengths = [0]
        num_intersections = [0]
        roundabouts = [0]
        traffic_signals = [0]
        stop_signs = [0]
        countries = [infer_country_from_city("Xiong'an")]

        # 遍历读取 CSV
        for filename in os.listdir(output_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(output_dir, filename)
                if not os.path.exists(filepath):
                    continue

                city_name = filename.replace('_road_mileage.csv', '').replace('_', ' ').strip()

                # 将 Xiongan 改为 Xiong'an
                if city_name == 'Xiongan':
                    city_name = "Xiong'an"

                if city_name == "Sanya":
                    temp_city_name = "Sanya, China"
                elif city_name == "Yangquan":
                    temp_city_name = "Yangquan, China"
                else:
                    temp_city_name = city_name

                df = pd.read_csv(filepath)
                if 'road_type' not in df.columns or 'value' not in df.columns:
                    continue

                df.set_index('road_type', inplace=True)

                total_length = safe_get_value(df, 'total')
                total_inter = safe_get_value(df, 'num_inter')
                num_roundabouts = int(safe_get_value(df, 'roundabouts'))
                num_traffic_signals = int(safe_get_value(df, 'traffic_signals'))
                num_stop_signs = int(safe_get_value(df, 'stop_signs'))

                if 'total' not in df.index or 'num_inter' not in df.index or \
                        'roundabouts' not in df.index or 'traffic_signals' not in df.index or 'stop_signs' not in df.index:
                    continue

                country = infer_country_from_city(temp_city_name)

                cities.append(city_name)
                road_lengths.append(total_length)
                num_intersections.append(total_inter)
                roundabouts.append(num_roundabouts)
                traffic_signals.append(num_traffic_signals)
                stop_signs.append(num_stop_signs)
                countries.append(country)

        # 排序
        sorted_indices = sorted(range(len(cities)), key=lambda i: road_lengths[i])
        cities = [cities[i] for i in sorted_indices]
        road_lengths = [road_lengths[i] for i in sorted_indices]
        num_intersections = [num_intersections[i] for i in sorted_indices]
        roundabouts = [roundabouts[i] for i in sorted_indices]
        traffic_signals = [traffic_signals[i] for i in sorted_indices]
        stop_signs = [stop_signs[i] for i in sorted_indices]
        countries = [countries[i] for i in sorted_indices]

        x = np.arange(len(cities))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(22, 8))

        # 背景条纹
        for i in range(len(cities)):
            if i % 2 == 0:
                ax1.axvspan(i - 0.5, i + 0.5, facecolor='gray', alpha=0.3, zorder=0)

        # 定义颜色
        color_road = '#4C72B0'  # 对应左轴
        color_inter = '#C44E52'  # 对应右轴


        class ZeroAwareScalarFormatter(ScalarFormatter):
            def __call__(self, x, pos=None):
                if np.isclose(x, 0, atol=1e-12):
                    return "0"
                return super().__call__(x, pos)


        # Left Y-axis: covered road mileage (blue)
        bars1 = ax1.bar(x - width / 2, road_lengths, width, color=color_road, label='Covered road mileage (km)')
        ax1.set_ylabel('Covered road mileage (km)', color=color_road)
        ax1.tick_params(axis='y', labelcolor=color_road)

        formatter = ZeroAwareScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax1.yaxis.set_major_formatter(formatter)

        # Right Y-axis: covered population (red)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, num_intersections, width, color=color_inter, label='Covered intersections')
        ax2.set_ylabel('Covered intersections', color=color_inter)
        ax2.tick_params(axis='y', labelcolor=color_inter)

        formatter2 = ZeroAwareScalarFormatter(useOffset=False, useMathText=True)
        formatter2.set_scientific(True)
        formatter2.set_powerlimits((0, 0))
        ax2.yaxis.set_major_formatter(formatter2)

        # 设置 X 轴标签颜色
        unique_countries = list(dict.fromkeys(countries))
        country_colors = {}
        for country in unique_countries:
            if country in COUNTRY_COLOR:
                country_colors[country] = COUNTRY_COLOR[country]
            else:
                cmap = plt.get_cmap('Dark2')
                country_colors[country] = cmap(len(country_colors) % cmap.N)

        ax1.set_xticks(x)
        ax1.set_xticklabels(cities, rotation=90, ha='center')
        for lbl, ctry in zip(ax1.get_xticklabels(), countries):
            lbl.set_color(country_colors.get(ctry, '#333333'))

        # 主图例
        legend_elements = [
            Patch(facecolor=color_road, edgecolor='k', label='Covered road mileage (km)'),
            Patch(facecolor=color_inter, edgecolor='k', label='Covered intersections')
        ]

        # 创建第一个图例
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

        # 统计数据图例
        total_road_len = sum(road_lengths)
        total_inter_num = sum(num_intersections)


        # 格式化数字函数 (直接使用你的逻辑)
        def format_number_str(num):
            if num >= 1e6:
                # 百万级
                return f"{num:,.0f}"
            elif num >= 1e3:
                # 千级，保留一位小数或整数
                return f"{num:,.1f}" if num % 1 != 0 else f"{int(num):,}"
            else:
                return f"{num:.1f}"


        road_str = f"{format_number_str(total_road_len)} km"
        inter_str = f"{format_number_str(total_inter_num)}"

        # 创建用于显示的 Handles (色块) 和 Labels (数值)
        total_handles = [
            Patch(facecolor=color_road, edgecolor='k', label=road_str),
            Patch(facecolor=color_inter, edgecolor='k', label=inter_str)
        ]

        # 创建第二个图例
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
        fig.savefig(Path(output_dir).parent / "Road_Calc.png", dpi=300, bbox_inches='tight')
        plt.close()