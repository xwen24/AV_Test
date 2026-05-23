ROAD_TYPES = [
    'motorway',
    'trunk',
    'primary',
    'secondary',
    'tertiary'
]


COUNTRY_COLOR = {
    '中国': '#E63946',         # 鲜红色 — 高饱和，醒目
    'United States': '#2B2B2B',  # 深炭黑（略带灰，避免纯黑）
    'Deutschland': '#FFD166',  # 明亮黄色 — 与深色形成强烈对比
    'الإمارات العربية المتحدة': '#9B5DE5',  # 霓虹紫 — 高辨识度
    '대한민국': '#06D6A0'      # 青绿色 — 清新且独特
}


CHN_CITY_NAME = {
    "Guangzhou": "广州市",
    "Dongguan": "东莞市",
    "Shenzhen": "深圳市",
    "Foshan": "佛山市",
    "Jiangmen": "江门市"
}


THRESHOLD = 500000


SKIP_INACCESSIBLE_IDS = {
    "894",  # No data elements in server response. Check query location/filters and log.
    "8486",  # No data elements in server response. Check query location/filters and log.
    "2508"  # No data elements in server response. Check query location/filters and log.
}


# refer to https://doi.org/10.1016/j.isprsjprs.2022.12.021
CLIMATE_THRESHOLDS = {
    'extreme_prep_thres' : 20,
    'max_temp_thres' : 40,
    'min_temp_thres' : -10
}


POI_TAGS = {
    'amenity': True,
    'leisure': True,
    'office': True,
    'public_transport': True,
    'railway': True,
    'shop': True,
    'tourism': True,
}


POI_STATION_TAGS = {
    'railway': ['station', 'halt', 'tram_stop'],          # 覆盖火车、大部分地铁
    'station': ['subway', 'light_rail', 'train'],     # 覆盖明确的地铁/轻轨
    'public_transport': ['station'],         # 覆盖符合新标准的站点
}


LOG_TRANSFORM_VARIABLES = ['gdp', 'pop_density',
                           'annual_prep', 'snowfall',
                           'max_temp', 'min_temp',
                           'poi_richness']


# unitless
LABELS = {
    "gdp": r"GDP per capita (US$)",
    "gdp_log": r"log GDP per capita (US$)",
    "gdp_sum": r"Total GDP (US$)",
    "pop_density": r"Population density (/km$^2$)",
    "pop_density_log": r"log Population density (/km$^2$)",
    "pop_size": "Population size",
    "ntl_mean": r"NTL intensity (nW cm$^{-2}$ sr$^{-1}$)",
    "ntl_mean_log": r"log NTL intensity (nW cm$^{-2}$ sr$^{-1}$)",
    "annual_prep": "Annual rainfall (mm)",
    "annual_prep_log": "log Annual rainfall (mm)",
    "extreme_prep": "Annual days with rainfall>20mm/d",
    "snowfall": "Annual snowfall (mm)",
    "snowfall_log": "log Annual snowfall (mm)",
    "max_temp": "Annual days above 40°C",
    "max_temp_log": "log Annual days above 40°C",
    "min_temp": "Annual days below -10°C",
    "min_temp_log": "log Annual days below -10°C",
    "slope": "Slope (°)",
    "entropy": "Road network orientation entropy",
    "road_density": "Road density (km/km$^2$)",
    "sinuosity": "Road sinuosity",
    "ratio_complex": "Complex intersection ratio (%)",
    "dist_to_station": "Distance to nearest station (km)",
    "poi_richness": "POI richness",
    "poi_richness_log": "log POI richness",
    "poi_entropy": "POI Shannon entropy"
}

LABELS_UNITLESS = {
    "gdp": r"GDP per capita",
    "gdp_log": r"log GDP per capita",
    "gdp_sum": r"Total GDP",
    "pop_density": r"Population density",
    "pop_density_log": r"log Population density",
    "pop_size": "Population size",
    "ntl_mean": r"NTL intensity",
    "ntl_mean_log": r"log NTL intensity",
    "annual_prep": "Annual rainfall",
    "annual_prep_log": "log Annual rainfall",
    "extreme_prep": "Annual days with rainfall>20mm/d",
    "snowfall": "Annual snowfall",
    "snowfall_log": "log Annual snowfall",
    "max_temp": "Annual days above 40°C",
    "max_temp_log": "log Annual days above 40°C",
    "min_temp": "Annual days below -10°C",
    "min_temp_log": "log Annual days below -10°C",
    "slope": "Slope",
    "entropy": "Road network orientation entropy",
    "road_density": "Road density",
    "sinuosity": "Road sinuosity",
    "ratio_complex": "Complex intersection ratio (%)",
    "dist_to_station": "Distance to nearest station",
    "poi_richness": "POI richness",
    "poi_richness_log": "log POI richness",
    "poi_entropy": "POI Shannon entropy"
}