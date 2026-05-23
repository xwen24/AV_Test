import os
import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scienceplots
import seaborn as sns
from scipy.stats import spearmanr, linregress, ks_2samp
from utilities.hyperparameters import COUNTRY_COLOR, THRESHOLD, LABELS, LABELS_UNITLESS
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from City_Dist_3 import read_gmt_border
china_file = "Data/CHN_Map/CN-border-L1.gmt"
from Global_Comp_6 import clean_df
import country_converter as coco
import textwrap


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

def _aggregate_df(df, metric, id_col):
    if id_col not in df.columns:
        candidates = [c for c in df.columns if id_col.lower() in c.lower()]
        if candidates:
            print(f"Warning: Exact column '{id_col}' not found. Using '{candidates[0]}' instead.")
            id_col = candidates[0]
        else:
            raise KeyError(f"Required column '{id_col}' not found. Available: {list(df.columns)}")

    # Case 1: 原子指标
    if metric in df.columns and metric not in ["snowfall", "max_temp", "min_temp"]:
        df_out = df[[id_col, metric]].copy()
        df_out[metric] = pd.to_numeric(df_out[metric], errors='coerce')
        return df_out.rename(columns={id_col: 'id'})

    # Case 2: 需要聚合的指标
    metric_cols = [col for col in df.columns if col != id_col and metric in col.lower()]

    if not metric_cols:
        if metric in df.columns:
            metric_cols = [metric]
        else:
            print(f"Warning: No columns found for metric '{metric}' in dataframe. Skipping.")
            return pd.DataFrame(columns=['id', metric])

    df_subset = df[[id_col] + metric_cols].copy()

    # 转换为数值型，无法转换的变为 NaN
    numeric_data = df_subset[metric_cols].apply(pd.to_numeric, errors='coerce')

    # 删除全为 NaN 的行
    mask = numeric_data.notna().any(axis=1)
    df_subset = df_subset[mask].copy()
    numeric_data = numeric_data[mask]

    if df_subset.empty:
        return pd.DataFrame(columns=['id', metric])

    # 根据指标类型选择聚合方式
    if metric == "max_temp":
        df_subset[metric] = numeric_data.max(axis=1)
    elif metric == "min_temp":
        df_subset[metric] = numeric_data.min(axis=1)
    elif metric in ["precip_intensity", "snowfall"]:
        df_subset[metric] = numeric_data.mean(axis=1)
    else:
        df_subset[metric] = numeric_data.mean(axis=1)

    return df_subset[[id_col, metric]].rename(columns={id_col: 'id'})


def load_and_process_data(metrics, data_dir, cols_to_log=None):
    cols_to_log = cols_to_log if cols_to_log else []

    def _load_group_data(group_prefix, id_col_name):
        combined_df = None

        for target_metric in metrics:
            # --- 步骤 1: 确定数据源文件名和是否需要转换 ---
            source_metric = target_metric  # 默认源文件就是 metric 本身
            needs_log = False

            for raw_col in cols_to_log:
                if target_metric == f"{raw_col}_log":
                    source_metric = raw_col
                    needs_log = True
                    break

            # --- 步骤 2: 加载源文件 ---
            path = os.path.join(data_dir, f'{group_prefix}_{source_metric}.csv')

            if not os.path.exists(path):
                print(f"Warning: File not found {path}, skipping metric '{target_metric}'...")
                continue

            df = pd.read_csv(path)

            try:
                # --- 步骤 3: 聚合数据 ---
                df_agg = _aggregate_df(df, source_metric, id_col=id_col_name)
            except Exception as e:
                print(f"Error processing {source_metric} for target {target_metric}: {e}")
                continue

            if df_agg.empty:
                continue

            # --- 步骤 4: 执行 Log 转换 ---
            if needs_log:
                if source_metric in df_agg.columns:
                    # 执行 log1p 转换
                    df_agg[target_metric] = np.log1p(df_agg[source_metric])
                    df_agg = df_agg[['id', target_metric]]
                else:
                    print(f"Error: Aggregated dataframe missing column '{source_metric}'")
                    continue

            # --- 步骤 5: 合并 ---
            if combined_df is None:
                combined_df = df_agg
            else:
                combined_df = combined_df.merge(df_agg, on='id', how='inner')

        return combined_df

    print("Loading Served Data...")
    df_accessible = _load_group_data('av_accessible', 'city')

    print("Loading Unserved Data...")
    df_inaccessible = _load_group_data('av_inaccessible', 'ID_UC_G0')

    if df_accessible is None or df_inaccessible is None:
        print("Error: Failed to load sufficient data.")
        return None

    df_accessible['group'] = 'AV-served'
    df_inaccessible['group'] = 'AV-unserved'

    print(f"Loaded: Served ({len(df_accessible)}), Unserved ({len(df_inaccessible)})")

    return pd.concat([df_accessible, df_inaccessible], ignore_index=True)


def plot_correlation(combined_df, metrics, save_path):
    # 1. Filter valid columns
    valid_metrics = [m for m in metrics if m in combined_df.columns]
    if not valid_metrics:
        print("No valid metrics found.")
        return None, None, None

    friendly_metrics = [LABELS_UNITLESS.get(m, m) for m in valid_metrics]
    metric_map = dict(zip(valid_metrics, friendly_metrics))
    df_renamed = combined_df.rename(columns=metric_map)[friendly_metrics].copy()

    # Drop rows with any missing values in selected columns
    df_clean = df_renamed.dropna()

    if df_clean.shape[0] == 0:
        print("No data after dropping NaNs.")
        return None, None, None

    # 2. Correlation Heatmap
    corr_matrix = df_clean.corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig1, ax1 = plt.subplots(figsize=(12, 10))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        ax=ax1,
        annot=True,
        annot_kws={"size": 16, "weight": "bold"},
        fmt=".2f",
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=1,
        linecolor='white',
        cbar_kws={"shrink": .75, "aspect": 20}
    )

    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    # Manually set colorbar label and font size
    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Spearman Correlation", fontsize=20)

    # Set tick labels manually
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', weight='medium', fontsize=20)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, weight='medium', fontsize=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        corr_path = os.path.join(save_path, "correlation.png")
        fig1.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"Save to: {corr_path}")

    # 3. VIF Calculation
    X = df_clean.copy()

    # Ensure all columns are numeric
    if not all(np.issubdtype(X[col].dtype, np.number) for col in X.columns):
        raise ValueError("All selected metrics must be numeric for VIF.")

    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
        except np.linalg.LinAlgError:
            vif = np.nan
        vif_data.append({'Metric': col, 'VIF': vif})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=True)

    # 4. VIF Bar Plot — use pure matplotlib styling
    fig2, ax2 = plt.subplots(figsize=(12, 6))

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

    bars = ax2.barh(vif_df['Metric'], vif_df['VIF'], color=colors, alpha=0.8, edgecolor='none', height=0.6)

    # Reference lines
    ax2.axvline(x=10, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add labels for thresholds — 增大字体，加粗
    trans = ax2.get_xaxis_transform()
    ax2.text(10, 1.01, 'Threshold=10', transform=trans, color='#e74c3c', ha='center', weight='bold', fontsize=14)

    # X轴标签 — 设置字体大小
    ax2.tick_params(axis='x', labelsize=16)
    ax2.set_xlabel('Variance inflation factor (VIF)', fontsize=16, labelpad=10, fontweight='medium')

    # Y轴标签（指标名称）— 设置字体更大、加粗
    ax2.set_yticklabels(vif_df['Metric'], fontsize=16, rotation=0)

    # Clean up spines — 显示所有边框
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    ax2.grid(axis='x', linestyle=':', alpha=0.6)

    # Adjust x-limit
    max_vif = vif_df['VIF'].max()
    if pd.notna(max_vif):
        ax2.set_xlim(0, max(max_vif * 1.15, 12))

    # Annotate bars
    for bar, vif in zip(bars, vif_df['VIF']):
        width = bar.get_width()
        if pd.notna(vif):
            label_x_pos = width + (ax2.get_xlim()[1] * 0.01)
            ax2.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                     f'{vif:.1f}', va='center', fontsize=14, color='#333333', fontweight='bold')

    plt.tight_layout()

    if save_path:
        vif_path = os.path.join(save_path, "vif.png")
        fig2.savefig(vif_path, dpi=300, bbox_inches='tight')
        print(f"Save to: {vif_path}")

    plt.close(fig1)
    plt.close(fig2)

    return fig1, fig2, vif_df


def plot_metric_vs_metric(combined_df, metric1, metric2, save_path, auto_zoom=True):
    # === 0. 断轴配置 (Configuration) ===
    # 格式: {(metric1, metric2): (break_start, break_end, ratio_left, ratio_right)}
    BROKEN_X_CONFIG = {
        ('annual_prep', 'snowfall'): (45, 100, 3, 1),
    }

    # 检查是否触发断轴
    is_broken = (metric1, metric2) in BROKEN_X_CONFIG
    break_params = BROKEN_X_CONFIG.get((metric1, metric2))

    # === 1. 数据准备 ===
    required_cols = {'id', 'group', metric1, metric2}
    if not required_cols.issubset(combined_df.columns):
        missing = required_cols - set(combined_df.columns)
        raise ValueError(f"Missing columns in combined_df: {missing}")

    plot_df = combined_df[['id', 'group', metric1, metric2]].copy()
    plot_df = plot_df.dropna(subset=[metric1, metric2])
    plot_df['Accessibility'] = plot_df['group'].replace({
        'AV-served': 'Served',
        'AV-unserved': 'Unserved'
    })

    # Add country info
    country_map = {}
    city_country_file = 'src/Results/City_Dist/city_geocodes.csv'
    try:
        if os.path.exists(city_country_file):
            country_df = pd.read_csv(city_country_file)
            if 'City' in country_df.columns and 'Country' in country_df.columns:
                country_map = dict(zip(country_df['City'], country_df['Country']))
    except Exception as e:
        print(f"⚠️ 无法加载城市-国家映射文件: {e}")

    plot_df['Country'] = plot_df.apply(
        lambda row: country_map.get(row['id'], 'Unknown')
        if row['Accessibility'] == 'Served' else 'N/A',
        axis=1
    )

    # Compute quadrant boundaries
    x = plot_df[metric2]
    y = plot_df[metric1]
    mean_x, mean_y = x.mean(), y.mean()

    # Compute correlation
    r_total = spearmanr(x, y)[0] if len(plot_df) > 1 else 0

    xlabel = LABELS.get(metric2, metric2)
    ylabel = LABELS.get(metric1, metric1)

    # === 2. 绘图设置 ===
    country_color_english = {
        'China': COUNTRY_COLOR.get('中国', 'red'),
        'USA': COUNTRY_COLOR.get('United States', 'black'),
        'Germany': COUNTRY_COLOR.get('Deutschland', 'orange'),
        'UAE': COUNTRY_COLOR.get('الإمارات العربية المتحدة', 'purple'),
        'Korea': COUNTRY_COLOR.get('대한민국', 'green')
    }
    av_unserved_color = '#808080'
    country_marker_map = {'China': 's', 'Germany': '^', 'Korea': 'X', 'UAE': 'v', 'USA': 'o'}

    fig = plt.figure(figsize=(10, 8))

    # 定义统一的布局边界 (与 add_axes 的 rect [0.15, 0.15, 0.75, 0.80] 对应)
    layout_params = {
        'left': 0.15,
        'bottom': 0.15,
        'right': 0.90,
        'top': 0.95
    }

    if is_broken:
        break_start, break_end, ratio_l, ratio_r = break_params
        gs = fig.add_gridspec(1, 2, width_ratios=[ratio_l, ratio_r], wspace=0.05,
                              **layout_params)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        axes_list = [ax1, ax2]
        main_ax = ax1
    else:
        # rect = [left, bottom, width, height]
        rect = [
            layout_params['left'],
            layout_params['bottom'],
            layout_params['right'] - layout_params['left'],
            layout_params['top'] - layout_params['bottom']
        ]
        ax = fig.add_axes(rect)
        axes_list = [ax]
        main_ax = ax

    # === 3. 核心绘图逻辑 ===
    def plot_on_axis(target_ax):
        # 1. Unserved
        unserved_df = plot_df[plot_df['Accessibility'] == 'Unserved']
        target_ax.scatter(
            unserved_df[metric2], unserved_df[metric1],
            color=av_unserved_color, marker='o', s=100, alpha=0.2,
            edgecolor='white', label='Unserved'
        )
        # 2. Served
        served_df = plot_df[plot_df['Accessibility'] == 'Served']
        all_countries = served_df['Country'].dropna().unique()
        COUNTRY_DISPLAY_ORDER = ['China', 'Germany', 'Korea', 'UAE', 'USA']
        plot_countries = [c for c in COUNTRY_DISPLAY_ORDER if c in all_countries]

        for country in plot_countries:
            df_c = served_df[served_df['Country'] == country]
            color = country_color_english.get(country, '#2E86AB')
            marker = country_marker_map.get(country, 'o')
            target_ax.scatter(
                df_c[metric2], df_c[metric1],
                color=color, marker=marker, s=200, alpha=1,
                edgecolor='white', label=country
            )
        # 3. Mean lines
        target_ax.axvline(mean_x, color='#555555', linestyle='--', linewidth=2, alpha=0.8)
        target_ax.axhline(mean_y, color='#555555', linestyle='--', linewidth=2, alpha=0.8)

    for ax_curr in axes_list:
        plot_on_axis(ax_curr)

    # === 4. 坐标轴范围与断轴处理 ===
    y_min_p, y_max_p = np.percentile(y, 1), np.percentile(y, 99)
    y_span = y_max_p - y_min_p
    y_bottom, y_top = y_min_p - y_span * 0.15, y_max_p + y_span * 0.15
    if y.min() >= 0 and y_bottom < 0: y_bottom = -y_top * 0.02

    if is_broken:
        ax1, ax2 = axes_list
        break_start, break_end, _, _ = break_params
        ax1.set_ylim(bottom=y_bottom, top=y_top)

        # X轴范围
        x_min_p = np.percentile(x[x < break_start], 1) if len(x[x < break_start]) > 0 else x.min()
        x_left_limit = x_min_p - (break_start - x_min_p) * 0.1
        if x.min() >= 0 and x_left_limit < 0: x_left_limit = -break_start * 0.02
        ax1.set_xlim(x_left_limit, break_start)

        x_max_p = np.percentile(x[x > break_end], 99) if len(x[x > break_end]) > 0 else x.max()
        x_right_limit = x_max_p + (x_max_p - break_end) * 0.1
        ax2.set_xlim(break_end, x_right_limit)

        # 彻底清除断轴处的刻度
        # 1. 隐藏中间的脊柱(Spines)
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        # 2. 禁用断轴边缘的刻度线 (Tick Marks)
        ax1.tick_params(axis='y', right=False, labelright=False)
        ax2.tick_params(axis='y', left=False, labelleft=False)

        # 如果顶部或底部有刻度刚好落在断裂处，也可以尝试移除它们
        # 这里我们确保 Y 轴刻度只在最左侧显示
        ax1.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks_position('none')

        # 绘制平行的斜杠
        d = 0.015


        def draw_slashes(ax, x_pos_axes):
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False,
                          marker=[(-1, -1), (1, 1)], markersize=12, linestyle='none')
            ax.plot(x_pos_axes, 0, **kwargs)
            ax.plot(x_pos_axes, 1, **kwargs)


        draw_slashes(ax1, 1)  # 左图右边缘
        draw_slashes(ax2, 0)  # 右图左边缘

    else:
        if auto_zoom:
            x_min_p, x_max_p = np.percentile(x, 1), np.percentile(x, 99)
            x_span = x_max_p - x_min_p
            x_left, x_right = x_min_p - x_span * 0.15, x_max_p + x_span * 0.15
            if x.min() >= 0 and x_left < 0: x_left = -x_right * 0.02
            main_ax.set_xlim(left=x_left, right=x_right)
            main_ax.set_ylim(bottom=y_bottom, top=y_top)

    # === 5. 标签与刻度优化 ===
    main_ax.set_ylabel(ylabel, fontsize=24)
    if is_broken:
        fig.text(0.5, 0.04, xlabel, ha='center', fontsize=24)
        # 减少刻度数量
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    else:
        main_ax.set_xlabel(xlabel, fontsize=24)
        for ax_curr in axes_list:
            ax_curr.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    for ax_curr in axes_list:
        ax_curr.tick_params(axis='both', which='major', labelsize=20)

    # === 6. 图例 ===
    served_df = plot_df[plot_df['Accessibility'] == 'Served']
    all_countries = served_df['Country'].dropna().unique()
    plot_countries = [c for c in ['China', 'Germany', 'Korea', 'UAE', 'USA'] if c in all_countries]

    legend_elements = [
        mlines.Line2D([0], [0], color='none', label=f'Spearman r = {r_total:.2f}'),
        mlines.Line2D([0], [0], color='none', label=' '),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=av_unserved_color, markersize=16,
                      label='Unserved'),
        mlines.Line2D([0], [0], color='none', label=r'$\bf{Served\ by\ Country:}$')
    ]
    for country in plot_countries:
        color = country_color_english.get(country, '#2E86AB')
        marker = country_marker_map.get(country, 'o')
        legend_elements.append(
            mlines.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=16, label=country))

    axes_list[-1].legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='gray',
                         framealpha=1, prop={'size': 20})

    # === 7. 科学计数法 ===
    sci_indicators = ['gdp', 'gdp_sum', 'pop_density', 'pop_size', 'annual_prep']
    forced_powers = {'gdp': 4, 'pop_density': 4}

    main_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))
    if metric1 in sci_indicators:
        offset_str = ""
        if metric1 in forced_powers:
            power = forced_powers[metric1]
            main_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 10 ** power:.1f}'))
            offset_str = f'$\\times 10^{{{power}}}$'
        else:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            main_ax.yaxis.set_major_formatter(formatter)
            fig.canvas.draw()
            offset_str = main_ax.yaxis.get_offset_text().get_text()

        if offset_str:
            main_ax.yaxis.offsetText.set_visible(False)
            main_ax.text(0, 1, offset_str, transform=main_ax.transAxes, ha='left', va='bottom', fontsize=22)

    for ax_curr in axes_list:
        if metric2 in sci_indicators:
            offset_str = ""
            if metric2 in forced_powers:
                power = forced_powers[metric2]
                ax_curr.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 10 ** power:.1f}'))
                offset_str = f'$\\times 10^{{{power}}}$'
            else:
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                ax_curr.xaxis.set_major_formatter(formatter)
                fig.canvas.draw()
                offset_str = ax_curr.xaxis.get_offset_text().get_text()

            if offset_str and ax_curr == axes_list[-1]:
                ax_curr.xaxis.offsetText.set_visible(False)
                ax_curr.text(1, 0, offset_str, transform=ax_curr.transAxes, ha='left', va='bottom', fontsize=22)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Save to: {save_path}")
    plt.close()


def plot_radar_chart(combined_df, metrics, save_path, metric_directions):
    # 1. Define Directionality
    valid_metrics = [m for m in metrics if m in combined_df.columns]
    if not valid_metrics:
        print("No valid metrics for radar chart.")
        return

    # 2. Calculate Normalization Bounds based on BENCHMARK
    benchmark_df = combined_df[combined_df['group'] == 'AV-unserved']
    if benchmark_df.empty:
        benchmark_df = combined_df

    bounds = {}
    for m in valid_metrics:
        # Check if column is numeric, otherwise skip or handle error
        if not np.issubdtype(benchmark_df[m].dtype, np.number):
            continue

        clean_data = benchmark_df[m].dropna()
        if clean_data.empty:
            bounds[m] = (0, 1)  # Default fallback
            continue

        p5 = np.percentile(clean_data, 5)
        p95 = np.percentile(clean_data, 95)
        bounds[m] = (p5, p95)

    # 3. Normalize Data
    df_norm = combined_df.copy()
    for m in valid_metrics:
        if m not in bounds: continue

        p5, p95 = bounds[m]
        denom = p95 - p5 if p95 != p5 else 1.0
        is_positive_direction = metric_directions.get(m, True)

        if is_positive_direction:
            df_norm[m] = (combined_df[m] - p5) / denom
        else:
            df_norm[m] = (p95 - combined_df[m]) / denom

        df_norm[m] = df_norm[m].clip(0, 1)

    # 4. Aggregate Data
    summary = df_norm.groupby('group')[valid_metrics].mean().reset_index()

    # 5. Prepare Plot Data
    # Wrap long labels to avoid them taking too much horizontal space
    categories = [textwrap.fill(LABELS_UNITLESS.get(m, m), width=15) for m in valid_metrics]
    N = len(categories)

    # Calculate angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    styles = {
        'AV-unserved': {'color': '#377eb8', 'label': 'AV-unserved'},
        'AV-served': {'color': '#e41a1c', 'label': 'AV-served'}
    }

    # Plot each group
    for group in ['AV-unserved', 'AV-served']:
        if group not in summary['group'].values:
            continue

        values = summary.loc[summary['group'] == group, valid_metrics].values.flatten().tolist()
        values += values[:1]

        style = styles[group]
        ax.plot(angles, values, linestyle='solid', label=style['label'], color=style['color'], linewidth=2)
        ax.fill(angles, values, color=style['color'], alpha=0.15)

    # 6. Styling the Chart
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set X-ticks (Angles)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    ax.tick_params(axis='x', pad=10)

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        x, y = label.get_position()

        # Convert angle to degrees for easier logic (0 is top, 90 is right, etc. due to offset)
        angle_deg = np.degrees(angle)

        # Determine alignment based on position in the circle
        if angle == 0:
            lab_ha = 'center'
            lab_va = 'bottom'
        elif 0 < angle < pi:
            lab_ha = 'left'
            lab_va = 'center'
        elif angle == pi:
            lab_ha = 'center'
            lab_va = 'top'
        else:  # pi < angle < 2*pi
            lab_ha = 'right'
            lab_va = 'center'

        label.set_horizontalalignment(lab_ha)
        label.set_verticalalignment(lab_va)

        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1'))

        label.set_zorder(15)

    # Y-ticks (Radial)
    ax.set_rlabel_position(0)  # Move radial labels to the top line to avoid clutter
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1.0"], color="#555", size=20)
    plt.ylim(0, 1.0)

    # Grid and Spines
    ax.grid(True, color='#ccc', linestyle='--', alpha=0.7)
    ax.spines['polar'].set_color('#ddd')

    # Legend
    ax.legend(
        loc='upper left', bbox_to_anchor=(-0.4, 1.2),
        frameon=True, facecolor='white',
        edgecolor='gray', handlelength=2
    )

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Save to: {save_path}")
    plt.close()


def compute_dcg_topsis(
    df_final,
    metrics,
    metric_directions=None,
    weight_method="entropy",   # 'entropy' or 'equal'
    lower_p=5,
    upper_p=95,
    impute_strategy="mean",
    verbose=True
):
    """
    从 df_final[metrics] 计算 TOPSIS-based deployment_compatibility_gap。

    Returns
    -------
    gap : np.ndarray shape (n_samples,)
    debug : dict (weights, bounds, scaler, etc.)
    """
    if metric_directions is None:
        metric_directions = {}

    missing_cols = [c for c in metrics if c not in df_final.columns]
    if missing_cols:
        raise ValueError(f"数据中缺失以下指标列: {missing_cols}")

    imputer = SimpleImputer(strategy=impute_strategy)
    X = imputer.fit_transform(df_final[metrics].values)

    if verbose:
        print("正在进行缩尾处理 以处理极值...")

    lower_bounds = np.percentile(X, lower_p, axis=0)
    upper_bounds = np.percentile(X, upper_p, axis=0)
    X = np.clip(X, lower_bounds, upper_bounds)

    # 1. Min-Max 归一化
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # 2. 处理正负向指标
    X_processed = X_norm.copy()
    for i, col in enumerate(metrics):
        direction = metric_directions.get(col, 1)  # 使用 get 避免 KeyError
        if direction == -1:
            X_processed[:, i] = 1 - X_processed[:, i]
        elif direction != 1:
            raise ValueError(f"指标 {col} 的方向定义错误，必须是 1 或 -1")

    # 3. 权重计算
    epsilon = 1e-12
    if weight_method == "equal":
        weights = np.ones(len(metrics)) / len(metrics)
        if verbose:
            print("Using Equal Weights (TOPSIS).")
    elif weight_method == "entropy":
        P = np.abs(X_processed) + epsilon
        P = P / P.sum(axis=0, keepdims=True)

        n = P.shape[0]
        k = 1 / np.log(n) if n > 1 else 0.0

        if n > 1:
            entropy = -k * np.sum(P * np.log(P), axis=0)
            d = 1 - entropy
            weights = d / (d.sum() + epsilon)
        else:
            weights = np.ones(len(metrics)) / len(metrics)

        if verbose:
            print("Using Entropy Weights (TOPSIS).")
    else:
        raise ValueError(f"未知的 weight_method: {weight_method}，请选择 'entropy' 或 'equal'")

    if verbose:
        weight_dict = dict(zip(metrics, weights))
        for kk, vv in sorted(weight_dict.items(), key=lambda item: item[1], reverse=True):
            print(f"  {kk}: {vv:.4f}")

    # 4. 构建加权矩阵
    Z = X_processed * weights

    # 5. 确定正负理想解
    Z_plus = Z.max(axis=0)
    Z_minus = Z.min(axis=0)

    # 6. 计算距离
    D_plus = np.sqrt(((Z - Z_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z - Z_minus) ** 2).sum(axis=1))

    # 7. 计算贴近度并转 gap
    scores = D_minus / (D_plus + D_minus + epsilon)
    gap = 1 - scores

    debug = {
        "method": "topsis",
        "weights": weights,
        "weight_method": weight_method,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "scaler": scaler,
        "imputer": imputer,
        "X_processed": X_processed,  # 可用于复现
    }
    return gap, debug


def compute_dcg_vikor(
    df_final,
    metrics,
    metric_directions=None,
    weight_method="entropy",   # 'entropy' or 'equal'
    lower_p=5,
    upper_p=95,
    impute_strategy="mean",
    v=0.5,
    verbose=True
):
    """
    从 df_final[metrics] 计算 VIKOR-based deployment_compatibility_gap (Q_norm)。

    Returns
    -------
    gap : np.ndarray shape (n_samples,)
    debug : dict
    """
    if metric_directions is None:
        metric_directions = {}

    missing_cols = [c for c in metrics if c not in df_final.columns]
    if missing_cols:
        raise ValueError(f"数据中缺失以下指标列: {missing_cols}")

    imputer = SimpleImputer(strategy=impute_strategy)
    X = imputer.fit_transform(df_final[metrics].values)

    if verbose:
        print("正在进行缩尾处理 以处理极值...")

    lower_bounds = np.percentile(X, lower_p, axis=0)
    upper_bounds = np.percentile(X, upper_p, axis=0)
    X = np.clip(X, lower_bounds, upper_bounds)

    # 1. Min-Max 标准化
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # 2. 统一指标方向：转换为"效益型"（越大越好）
    X_processed = X_norm.copy()
    for i, col in enumerate(metrics):
        direction = metric_directions.get(col, 1)
        if direction == -1:
            X_processed[:, i] = 1 - X_processed[:, i]
        elif direction != 1:
            raise ValueError(f"指标 {col} 的方向定义错误，必须是 1（效益型）或 -1（成本型）")

    # 3. 权重计算
    epsilon = 1e-12
    n, m = X_processed.shape

    if weight_method == "equal":
        weights = np.ones(m) / m
        if verbose:
            print("Using Equal Weights (VIKOR).")
    elif weight_method == "entropy":
        P = np.abs(X_processed) + epsilon
        P = P / P.sum(axis=0, keepdims=True)

        k = 1 / np.log(n) if n > 1 else 0.0
        if n > 1:
            entropy = -k * np.sum(P * np.log(P), axis=0)
            d = 1 - entropy
            weights = d / (d.sum() + epsilon)
        else:
            weights = np.ones(m) / m

        if verbose:
            print("Using Entropy Weights (VIKOR).")
    else:
        raise ValueError(f"未知的 weight_method: {weight_method}，请选择 'entropy' 或 'equal'")

    if verbose:
        weight_dict = dict(zip(metrics, weights))
        for kk, vv in sorted(weight_dict.items(), key=lambda item: item[1], reverse=True):
            print(f"  {kk}: {vv:.4f}")

    # 4. VIKOR 核心计算
    f_star = X_processed.max(axis=0)   # 最优值（效益型：越大越好）
    f_minus = X_processed.min(axis=0)  # 最劣值

    denom = f_star - f_minus
    denom = np.where(np.abs(denom) < epsilon, 1.0, denom)  # 常量指标处理

    regret = (f_star - X_processed) / denom  # shape: (n_samples, n_metrics)

    S = np.sum(weights * regret, axis=1)     # 群体效用损失
    R = np.max(weights * regret, axis=1)     # 最大遗憾

    S_star = S.min()
    S_minus = S.max()
    R_star = R.min()
    R_minus = R.max()

    denom_S = S_minus - S_star if (S_minus - S_star) > epsilon else epsilon
    denom_R = R_minus - R_star if (R_minus - R_star) > epsilon else epsilon

    Q = v * (S - S_star) / denom_S + (1 - v) * (R - R_star) / denom_R

    # 5. 归一化 Q 到 [0, 1]
    Q_min, Q_max = Q.min(), Q.max()
    if Q_max - Q_min < epsilon:
        Q_norm = np.full_like(Q, 0.5)
    else:
        Q_norm = (Q - Q_min) / (Q_max - Q_min)

    gap = Q_norm

    debug = {
        "method": "vikor",
        "weights": weights,
        "weight_method": weight_method,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "scaler": scaler,
        "imputer": imputer,
        "X_processed": X_processed,
        "v": v,
        "S": S,
        "R": R,
        "Q": Q,
    }
    return gap, debug


def plot_deployment_gap(combined_df, metrics, save_path, method,
                        metric_directions=None,
                        weight_method='entropy',  # 'entropy' 或 'equal'
                        accessible_coords_path='src/Results/City_Dist/city_geocodes.csv',
                        inaccessible_gpkg_path='Data/QGIS/GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg'):
    os.makedirs(save_path, exist_ok=True)

    # === 1. 坐标合并逻辑 ===
    print("正在合并城市坐标数据...")

    # 加载可部署城市坐标
    try:
        accessible_coords = pd.read_csv(accessible_coords_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {accessible_coords_path}")
        return

    # 加载不可部署城市坐标
    try:
        # 图层1: 获取几何（坐标）
        centroids_gdf = clean_df(gpd.read_file(inaccessible_gpkg_path, layer='UC_centroids'))
        if centroids_gdf.crs != "EPSG:4326":
            centroids_gdf = centroids_gdf.to_crs("EPSG:4326")

        # 只保留 ID 和 geometry
        centroids_gdf = centroids_gdf[['ID_UC_G0', 'geometry']].copy()
        centroids_gdf['ID_UC_G0'] = centroids_gdf['ID_UC_G0'].astype(str)

        # 图层2: 获取 city 和 country 属性
        attrs_gdf = clean_df(gpd.read_file(inaccessible_gpkg_path, layer='GHS_UCDB_THEME_GHSL_GLOBE_R2024A'))
        attrs_gdf['ID_UC_G0'] = attrs_gdf['ID_UC_G0'].astype(str)

        # 检查所需字段是否存在
        missing_attrs = []
        if 'GC_UCN_MAI_2025' not in attrs_gdf.columns:
            missing_attrs.append('GC_UCN_MAI_2025')
        if 'GC_CNT_GAD_2025' not in attrs_gdf.columns:
            missing_attrs.append('GC_CNT_GAD_2025')
        if missing_attrs:
            print(f"警告: GPKG 图层中缺失字段: {missing_attrs}")
            for col in missing_attrs:
                attrs_gdf[col] = None

        attr_cols = ['ID_UC_G0', 'GC_UCN_MAI_2025', 'GC_CNT_GAD_2025']
        attrs_subset = attrs_gdf[attr_cols].copy()

        # 合并几何与属性
        merged_gdf = centroids_gdf.merge(attrs_subset, on='ID_UC_G0', how='left')

        # 提取坐标和属性
        merged_gdf['Longitude'] = merged_gdf.geometry.x
        merged_gdf['Latitude'] = merged_gdf.geometry.y

        inaccessible_coords = merged_gdf[[
            'ID_UC_G0', 'Longitude', 'Latitude', 'GC_UCN_MAI_2025', 'GC_CNT_GAD_2025'
        ]].copy()
        inaccessible_coords.rename(columns={'GC_UCN_MAI_2025': 'City', 'GC_CNT_GAD_2025': 'Country'}, inplace=True)

    except Exception as e:
        print(f"错误: 加载 GPKG 失败 - {e}")
        return

    # 分离两组
    df_accessible = combined_df[combined_df['group'] == 'AV-served'].copy()
    df_inaccessible = combined_df[combined_df['group'] == 'AV-unserved'].copy()

    # 合并坐标 - Served
    df_accessible = df_accessible.merge(accessible_coords, left_on='id', right_on='City', how='left')

    # 合并坐标 - Unserved
    df_inaccessible['id_str'] = df_inaccessible['id'].astype(str)
    df_inaccessible = df_inaccessible.merge(
        inaccessible_coords, left_on='id_str', right_on='ID_UC_G0', how='left', suffixes=('', '_coord')
    )
    df_inaccessible.drop(columns=['id_str', 'ID_UC_G0'], inplace=True, errors='ignore')

    # 重新组合
    df_final = pd.concat([df_accessible, df_inaccessible], ignore_index=True)

    # 清洗缺失坐标
    df_final = df_final.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
    print(f"坐标合并完成，有效城市数量: {len(df_final)}")

    # === 2. 计算 Gap ===
    print(f"开始计算 Gap，使用方法: {method}, 权重模式: {weight_method}")

    missing_cols = [c for c in metrics if c not in df_final.columns]
    if missing_cols:
        raise ValueError(f"数据中缺失以下指标列: {missing_cols}")

    # 初始化结果列
    df_final['deployment_compatibility_gap'] = np.nan

    # 准备数据矩阵
    methods_needing_imputation = ['topsis', 'vikor']
    if method in methods_needing_imputation:
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(df_final[metrics].values)
    else:
        X = df_final[metrics].values
        if np.isnan(X).any():
            print(f"警告: 方法 {method} 的输入数据包含 NaN，建议加入自动填充列表。")

    print("正在进行缩尾处理 以处理极值...")

    # 计算每一列（每个指标）的上下界
    lower_p = 5
    upper_p = 95
    lower_bounds = np.percentile(X, lower_p, axis=0)
    upper_bounds = np.percentile(X, upper_p, axis=0)

    # 执行截断
    X = np.clip(X, lower_bounds, upper_bounds)

    if method == 'topsis':
        gap, debug = compute_dcg_topsis(
            df_final, metrics,
            metric_directions=metric_directions,
            weight_method=weight_method,
            verbose=True
        )
        df_final['deployment_compatibility_gap'] = gap

    elif method == 'vikor':
        gap, debug = compute_dcg_vikor(
            df_final, metrics,
            metric_directions=metric_directions,
            weight_method=weight_method,
            v=0.5,
            verbose=True
        )
        df_final['deployment_compatibility_gap'] = gap

    # 保存结果 (文件名增加权重模式标识)
    out_csv = f"{save_path}/deployment_compatibility_{method}_{weight_method}.csv"
    df_final.to_csv(out_csv, encoding='utf-8', index=False)
    print(f"Save to: {out_csv}")

    # 3. 绘图
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    land_color = '#f0f0f0'
    ocean_color = '#e8e8e8'
    border_color = '#aaaaaa'

    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=border_color, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor=border_color, zorder=1)

    # Draw China border
    if china_file:
        try:
            china_polys = read_gmt_border(china_file)
            for poly in china_polys:
                if len(poly) > 2:
                    polygon_fill = mpatches.Polygon(
                        poly, facecolor=land_color, edgecolor=border_color,
                        linewidth=0.5, linestyle='-', transform=ccrs.PlateCarree(), zorder=3
                    )
                    ax.add_patch(polygon_fill)
        except Exception as e:
            print(f"⚠️  China border drawing skipped: {e}")

    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    # --- 数据准备 ---
    bg_data = df_final[df_final['group'] == 'AV-unserved'].copy()
    fg_data = df_final[df_final['group'] == 'AV-served'].copy()

    # --- 颜色映射 ---
    vmin = np.percentile(bg_data['deployment_compatibility_gap'], 5)
    vmax = np.percentile(bg_data['deployment_compatibility_gap'], 95)
    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # --- 绘制散点 ---
    # 不可部署城市 (背景点)
    ax.scatter(
        bg_data['Longitude'], bg_data['Latitude'],
        c=bg_data['deployment_compatibility_gap'],
        s=30,
        cmap=cmap,
        norm=norm,
        edgecolor='none', alpha=0.8,
        transform=ccrs.PlateCarree(), zorder=8,
        label='AV-unserved cities'
    )

    # 可部署城市
    ax.scatter(
        fg_data['Longitude'], fg_data['Latitude'],
        s=120, c='gold', marker='*',
        edgecolor='black', linewidth=0.5,
        transform=ccrs.PlateCarree(), zorder=10,
        label='AV-served cities'
    )

    # --- 右下角直方图分布 ---
    ax_hist = inset_axes(ax, width="22.5%", height="15%", loc='lower left',
                         bbox_to_anchor=(0.04, 0.25, 1, 1),
                         bbox_transform=ax.transAxes)

    sns.histplot(bg_data['deployment_compatibility_gap'], bins=30, kde=True, ax=ax_hist,
                 color='#d62728', edgecolor='white', linewidth=0.5, alpha=0.7)

    avg_accessible_gap = fg_data['deployment_compatibility_gap'].mean()

    ax_hist.axvline(avg_accessible_gap, color='black', linestyle='--', linewidth=1.5)

    # --- 1. 单独画箭头 (精确控制起点和终点) ---
    arrow_start_x = 0.3
    arrow_start_y = 0.55

    ax_hist.annotate('',
                     xy=(avg_accessible_gap, ax_hist.get_ylim()[1] * 0.5),  # 箭头尖端 (终点)
                     xycoords='data',
                     xytext=(arrow_start_x, arrow_start_y),  # 箭头尾部 (起点)
                     textcoords='axes fraction',  # 起点使用相对坐标 (0-1)
                     arrowprops=dict(arrowstyle="->", color='black', linewidth=2))

    # --- 2. 单独写文字 ---
    ax_hist.text(0.2, 0.6,  # 文字位置
                 'AV-served \n mean',
                 transform=ax_hist.transAxes,  # 使用相对坐标系
                 horizontalalignment='center',  # 右对齐，这样文字结尾紧贴箭头起点
                 verticalalignment='center',
                 fontsize=14, color='black')

    # 动态标签
    ax_hist.set_xlabel(f'Deployment compatibility gap', fontsize=16)
    ax_hist.set_ylabel('Count', labelpad=0, fontsize=16)
    ax_hist.tick_params(axis='both', which='major', labelsize=14)
    ax_hist.patch.set_alpha(0.8)

    # --- Colorbar ---
    cax = ax.inset_axes([0.35, 0.12, 0.3, 0.03], transform=ax.transAxes)
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, orientation='horizontal'
    )
    cbar.set_label('Deployment compatibility gap', labelpad=8, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # --- Legend ---
    legend = ax.legend(
        loc='lower left', frameon=True,
        facecolor='white', edgecolor='gray', framealpha=1, markerscale=2, prop={'size': 18}
    )
    legend.get_frame().set_linewidth(0.5)

    out_png = f"{save_path}/deployment_compatibility_{method}_{weight_method}.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Save to: {out_png}")
    plt.close()

    return df_final


def plot_gap_drivers(df, target_metrics, save_path):
    # 确保 save_path 存在
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    valid_metrics = [m for m in target_metrics if m in df.columns]
    if not valid_metrics:
        print("No valid driver metrics found.")
        return

    for metric in valid_metrics:
        fig, ax = plt.subplots(figsize=(10, 8))

        # 准备数据
        sub_df = df.dropna(subset=[metric, 'deployment_compatibility_gap'])
        x = sub_df[metric]
        y = sub_df['deployment_compatibility_gap']

        # 1. 绘图 (Plotting)
        # A. 绘制散点 (Scatter)
        scatter_color = '#1f77b4'
        ax.scatter(x, y,
                   color=scatter_color,
                   edgecolor='white',
                   s=80,
                   alpha=0.7,
                   zorder=2)

        # B. 绘制回归线和置信区间
        sns.regplot(x=x, y=y, ax=ax, scatter=False,
                    color='black',
                    line_kws={'linewidth': 2},
                    ci=95)

        # 2. 计算统计量 (Statistics)
        corr, p_val = spearmanr(x, y)
        slope, intercept, _, _, _ = linregress(x, y)

        # 格式化 P值字符串
        if p_val < 0.001:
            p_str = "P < 0.001"
        else:
            p_str = f"P = {p_val:.3f}"

        # 标注文本
        stats_text = (
            f"$\\text{{Spearman }} r = {corr:.2f}$, {p_str}\n"
            f"Slope = {slope:.2f}, Int = {intercept:.2f}"
        )

        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=20,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

        # 3. 自定义图例 (Custom Legend)
        # 1. Cities (散点) - 手动创建一个带圆点的 Line2D
        cities_handle = mlines.Line2D([], [],
                                      color='white',  # 线条颜色设为白(即不显示线)
                                      marker='o',  # 圆点标记
                                      markerfacecolor=scatter_color,  # 填充色
                                      markeredgecolor='white',  # 边缘色
                                      markersize=14,
                                      label='Cities')

        # 2. Best fit (回归线) - 黑色实线
        line_handle = mlines.Line2D([], [], color='black', linewidth=2, label='Best fit')

        # 3. 95% CI (置信区间) - 黑色透明方块
        patch_handle = mpatches.Patch(color='black', alpha=0.2, label='95% CI')

        # 组合图例
        ax.legend(handles=[cities_handle, line_handle, patch_handle],
                  loc='lower right',
                  frameon=False,
                  fontsize=20)

        # 4. 坐标轴标签与其他设置
        x_label = LABELS.get(metric, metric)
        if x_label.lower() == 'beta':
            x_label = r'$\beta$'

        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel('Deployment compatibility gap', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)

        filename = f"{metric}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Save to: {full_path}")

        plt.close()


def plot_regional_inequality(df, save_path):
    # 1. 数据处理
    country_corrections = {
        'USA': 'United States',
        'UAE': 'United Arab Emirates',
        'Korea': 'South Korea',
        'México': 'Mexico'
    }
    df['Country'] = df['Country'].replace(country_corrections)

    cc = coco.CountryConverter()
    unique_countries = df['Country'].unique()

    # 转换大洲和区域
    continents_base = cc.convert(names=unique_countries, to='continent')
    regions_un = cc.convert(names=unique_countries, to='UNregion')

    final_continents = []
    if isinstance(continents_base, str):
        continents_base = [continents_base]
        regions_un = [regions_un]

    for continent, region in zip(continents_base, regions_un):
        if continent == 'America':
            if region == 'South America':
                final_continents.append('South America')
            else:
                final_continents.append('North America')
        else:
            final_continents.append(continent)

    country_to_continent = dict(zip(unique_countries, final_continents))
    df['Continent'] = df['Country'].map(country_to_continent)
    plot_df = df[df['Continent'] != 'not found'].copy()

    if plot_df.empty:
        print("Warning: No valid continents found.")
        return

    # 排序
    order = plot_df.groupby('Continent')['deployment_compatibility_gap'].median().sort_values().index

    # 2. 绘图设置
    # 创建画布
    fig, ax = plt.subplots(figsize=(19, 9))

    # A. 背景网格优化
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E0E0E0', linestyle='--', linewidth=1, alpha=0.8)

    # B. 配色方案
    palette = sns.color_palette("Set2", n_colors=len(order))

    # C. 绘制箱线图
    sns.boxplot(
        x='Continent',
        y='deployment_compatibility_gap',
        data=plot_df,
        order=order,
        palette=palette,
        linewidth=1.5,  # 边框线宽
        width=0.55,  # 箱体宽度
        fliersize=0,  # 隐藏异常点（由 stripplot 展示）
        ax=ax,
        boxprops=dict(alpha=0.9, edgecolor='#333333'),
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(color='#333333', linewidth=1.5),
        capprops=dict(color='#333333', linewidth=1.5)
    )

    # D. 绘制散点图 (Jitter)
    sns.stripplot(
        x='Continent',
        y='deployment_compatibility_gap',
        data=plot_df,
        order=order,
        color='#2b2b2b',  # 深灰色点
        alpha=0.4,  # 透明度，避免重叠时太黑
        size=7,
        jitter=0.2,
        edgecolor='white',  # 关键：给点加白色描边，使其更清晰
        linewidth=0.8,
        ax=ax
    )

    # 3. 边框与标签
    # E. 显式强化边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)  # 加粗边框
        spine.set_color('#333333')  # 深灰色，比纯黑更有质感

    # F. 优化 X 轴标签：加入样本量 (n=...)
    counts = plot_df['Continent'].value_counts()
    new_labels = [f"{c}\n(n={counts[c]})" for c in order]
    ax.set_xticklabels(new_labels, color='#333333', fontweight='medium')

    # G. Y 轴标签与刻度
    ax.set_ylabel('Deployment compatibility gap', labelpad=15, color='#333333')
    ax.set_xlabel('')
    plt.yticks(color='#333333')

    # 4. 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Save to: {save_path}")

    plt.close()


def plot_gap_contribution(combined_df, metrics, metric_directions, save_path, top_n=10):
    # 1. 数据准备：必须使用全量数据来确定"理想解"
    df_clean = combined_df.dropna(subset=metrics).copy()
    X = df_clean[metrics].values

    # 对每一列分别计算 5% 和 95% 分位数
    lower_bounds = np.percentile(X, 5, axis=0)
    upper_bounds = np.percentile(X, 95, axis=0)
    # 截断
    X = np.clip(X, lower_bounds, upper_bounds)

    # 2. 标准化
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # 3. 处理指标方向 (正向保持，负向反转)
    # 反转后：所有指标都是"越大越好"
    X_processed = X_norm.copy()
    for i, col in enumerate(metrics):
        direction = metric_directions.get(col, 1)
        if direction == -1:
            X_processed[:, i] = 1 - X_processed[:, i]

    # 4. 计算熵权
    epsilon = 1e-12
    P = X_processed.copy()
    P = np.abs(P) + epsilon
    P = P / P.sum(axis=0, keepdims=True)
    n = P.shape[0]
    k = 1 / np.log(n)
    entropy = -k * np.sum(P * np.log(P), axis=0)
    d = 1 - entropy
    weights = d / d.sum()

    # 5. 确定正理想解 (Z_plus)
    Z = X_processed * weights
    Z_plus = Z.max(axis=0)

    # 6. 计算每个城市、每个指标对"差距"的贡献
    # 贡献 = | 理想值 - 实际值 | (加权后)
    contribution_matrix = np.abs(Z_plus - Z)

    # 创建贡献度 DataFrame
    df_contrib = pd.DataFrame(contribution_matrix, columns=metrics)
    df_contrib['id'] = df_clean['id'].values
    df_contrib['group'] = df_clean['group'].values
    df_contrib['City'] = df_clean['City'].values
    df_contrib['Country'] = df_clean['Country'].replace({'South Korea': 'Korea', 'Taiwan': 'China'}).values
    df_contrib['deployment_compatibility_gap'] = df_clean['deployment_compatibility_gap'].values

    # 7. 筛选目标城市：未部署
    target_df = df_contrib[df_contrib['group'] == 'AV-unserved'].copy()

    # 解析 top_n 的数量
    n_cities = abs(top_n)  # 取绝对值用于筛选数量

    # 获取前 n_cities（最有潜力，Gap最小）和倒数前 n_cities（最无潜力，Gap最大）
    top_cities = target_df.nsmallest(n_cities, 'deployment_compatibility_gap')
    bottom_cities = target_df.nlargest(n_cities, 'deployment_compatibility_gap')

    # 定义一个辅助函数来处理绘图数据的排序和重命名，增加 sort_ascending 参数
    def prepare_plot_data(df, sort_ascending):
        df_copy = df.copy()
        # 拼接城市和国家，国家另起一行并加上括号
        df_copy['City_Label'] = df_copy['City'] + '\n(' + df_copy['Country'] + ')'

        plot_data = df_copy.set_index('City_Label')[metrics]
        plot_data['sum_dist'] = plot_data.sum(axis=1)

        # 根据传入的参数决定排序方式
        plot_data = plot_data.sort_values('sum_dist', ascending=sort_ascending)
        plot_data = plot_data.drop(columns=['sum_dist'])

        # 如果全局定义了 LABELS_UNITLESS，则重命名列
        if 'LABELS_UNITLESS' in globals():
            plot_data = plot_data.rename(columns=LABELS_UNITLESS)
        return plot_data

    # 左图：从上往下“由小到大”，即最上面是最小的，最下面是最大的 -> 数据需按从大到小排序 (ascending=False)
    plot_data_top = prepare_plot_data(top_cities, sort_ascending=False)
    # 右图：从上往下“由大到小”，即最上面是最大的，最下面是最小的 -> 数据需按从小到大排序 (ascending=True)
    plot_data_bottom = prepare_plot_data(bottom_cities, sort_ascending=True)

    # 8. 绘图：创建 1 行 2 列的画布
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))

    # 获取颜色
    num_metrics = len(plot_data_top.columns)
    cmap = cm.get_cmap('tab20', num_metrics)
    colors = [cmap(i) for i in range(num_metrics)]

    # 绘制左图
    plot_data_top.plot(kind='barh', stacked=True, ax=axes[0], color=colors, width=0.7, edgecolor='white', linewidth=0.5)
    axes[0].set_title(f'Top {abs(top_n)} cities with the lowest DCG', fontsize=32, pad=15)

    # 绘制右图
    plot_data_bottom.plot(kind='barh', stacked=True, ax=axes[1], color=colors, width=0.7, edgecolor='white', linewidth=0.5)
    axes[1].set_title(f'Bottom {abs(top_n)} cities with the highest DCG', fontsize=32, pad=15)

    # 添加 a 和 b 标签
    font_kwargs = {'fontsize': 55, 'fontweight': 'bold', 'fontfamily': 'Times New Roman'}

    # 在左图左上角添加 a
    axes[0].text(-0.15, 1.02, 'a', transform=axes[0].transAxes, **font_kwargs)

    # 在右图左上角添加 b
    axes[1].text(-0.15, 1.02, 'b', transform=axes[1].transAxes, **font_kwargs)

    # 9. 装饰与标注
    for ax in axes:
        ax.set_xlabel('Contribution to the deployment compatibility gap', fontsize=32, labelpad=10)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=28)
        ax.tick_params(axis='x', labelsize=28)

        # 移除子图自带的图例
        if ax.get_legend():
            ax.get_legend().remove()

        # 显式开启所有边框，并设置颜色和线宽
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(1)

        # 添加网格线
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.53, -0.17),
               ncol=4, frameon=True, facecolor='white', edgecolor='gray',
               framealpha=1, fontsize=32)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Save to: {save_path}")

    plt.close()


def plot_top_potential_cities(df, save_path):
    # 1. Filter Data: Only looking at undeployed cities
    df_target = df[df['group'] == 'AV-unserved'].copy()

    # 2. Sort and take top 20 (Smaller Gap is better)
    # We sort ascending=False here so that when plotting, the smallest value (best rank) appears at the top
    top_20 = df_target.nsmallest(20, 'deployment_compatibility_gap').sort_values('deployment_compatibility_gap', ascending=False)

    # Create labels: City (Country)
    top_20['label'] = top_20['City'] + " (" + top_20['Country'] + ")"

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(20, 12))

    # Draw lines (Lollipop stems)
    ax.hlines(y=top_20['label'], xmin=0, xmax=top_20['deployment_compatibility_gap'],
              color='skyblue', alpha=0.6, linewidth=2)

    # Draw points (Lollipop heads)
    ax.scatter(top_20['deployment_compatibility_gap'], top_20['label'],
               color='#d62728', s=100, alpha=1, zorder=3)

    # Decoration
    ax.set_xlabel('Deployment compatibility gap')
    # ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Get the maximum value to determine the x-axis limit
    max_val = top_20['deployment_compatibility_gap'].max()

    # Add numerical labels
    for i, (val, name) in enumerate(zip(top_20['deployment_compatibility_gap'], top_20['label'])):
        # We place the text slightly to the right of the point.
        # transform=ax.transData ensures coordinates are based on data values.
        ax.text(val, i, f" {val:.3f}", va='center', ha='left', color='black')

    # Adjust x-axis limits to create whitespace on the right for the text
    # We add about 10-15% padding to the maximum value
    ax.set_xlim(0, max_val * 1.15)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Save to: {save_path}")

    plt.close()


def plot_gap_cdf(df, save_path):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.20, 0.12, 0.75, 0.80])  # [left, bottom, width, height]

    # 颜色映射调整
    palette = {'AV-served': 'lightblue', 'AV-unserved': 'lightcoral'}

    # 绘制 CDF
    sns.ecdfplot(data=df, x='deployment_compatibility_gap', hue='group', palette=palette, linewidth=3, ax=ax)

    # 计算 KS 统计量
    acc = df[df['group'] == 'AV-served']['deployment_compatibility_gap']
    inacc = df[df['group'] == 'AV-unserved']['deployment_compatibility_gap']
    ks_stat, p_val = ks_2samp(acc, inacc)

    # 绘制统计文本
    p_text = "P < 0.010" if p_val < 0.01 else f"P = {p_val:.3f}"
    stats_label = f'{p_text}'

    ax.text(0.05, 0.95, stats_label,
            transform=ax.transAxes, verticalalignment='top')

    # 标签与轴设置
    ax.set_xlabel('Deployment compatibility gap')
    ax.set_ylabel('Cumulative probability')

    # 设置坐标轴范围 (可选，根据数据自动，但确保从 0 开始通常较好)
    ax.set_ylim(-0.02, 1.02)

    # 图例优化
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='AV-served'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='AV-unserved')
    ]

    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(0.01, 0.99),
              frameon=True,
              facecolor='white',
              edgecolor='gray',
              framealpha=1)

    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Save to: {save_path}")

    plt.close()


def plot_logistic_curve(df, save_path):
    df_model = df.dropna(subset=['deployment_compatibility_gap']).copy()
    df_model['target'] = df_model['group'].apply(lambda x: 1 if x == 'AV-served' else 0)

    X = df_model[['deployment_compatibility_gap']]
    y = df_model['target']

    # 拟合模型
    clf = LogisticRegression()
    clf.fit(X, y)

    # 预测曲线
    X_test = np.linspace(df_model['deployment_compatibility_gap'].min(), df_model['deployment_compatibility_gap'].max(), 300).reshape(
        -1, 1)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # 寻找阈值 (Prob = 0.5)
    threshold_idx = np.abs(y_prob - 0.5).argmin()
    threshold_val = X_test[threshold_idx][0]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.20, 0.12, 0.75, 0.80])  # [left, bottom, width, height] 保持边距一致

    # 绘制散点 (顶部和底部)
    ax.scatter(df_model[df_model['target'] == 1]['deployment_compatibility_gap'],
               [1] * len(df_model[df_model['target'] == 1]),
               color='cornflowerblue', alpha=0.6, marker='|', s=100, linewidths=2,
               label='_nolegend_')

    ax.scatter(df_model[df_model['target'] == 0]['deployment_compatibility_gap'],
               [0] * len(df_model[df_model['target'] == 0]),
               color='indianred', alpha=0.4, marker='|', s=100, linewidths=2, label='_nolegend_')

    # 绘制 S 曲线
    ax.plot(X_test, y_prob, color='black', linewidth=2, label='Logistic Fit')

    # 绘制辅助线
    ax.axvline(threshold_val, color='green', linestyle='--', linewidth=1.5)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

    # 自定义图例
    legend_elements = [
        mlines.Line2D([0], [0], color='cornflowerblue', lw=5, alpha=0.6, label='AV-served'),
        mlines.Line2D([0], [0], color='indianred', lw=5, alpha=0.4, label='AV-unserved'),
        mlines.Line2D([0], [0], color='black', lw=2, label='Logistic Fit'),
        mlines.Line2D([0], [0], color='green', linestyle='--', lw=1.5, label=f'Threshold ({threshold_val:.2f})')
    ]

    ax.set_xlabel('Deployment compatibility gap')
    ax.set_ylabel('Probability of AV Deployment')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(df_model['deployment_compatibility_gap'].min(), df_model['deployment_compatibility_gap'].max())

    ax.legend(handles=legend_elements, loc='center right', frameon=True,
              facecolor='white', edgecolor='gray', handlelength=2.5, prop={'size': 24})

    # 如果确实需要网格，取消下面这行的注释
    # ax.grid(True, linestyle=':', alpha=0.3)

    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Save to: {save_path}")

    plt.close()


def plot_variable_width_bar(df, method, save_path, indicator="deployment_compatibility_gap",
                            width_indicator=None, global_north_file='src/utilities/GN.csv',
                            min_width_threshold=1.5,
                            force_inside_countries=['Nigeria', 'France', 'Germany']):
    if force_inside_countries is None:
        force_inside_countries = []

    country_corrections = {
        'USA': 'United States',
        'UAE': 'United Arab Emirates',
        'Korea': 'South Korea',
        'México': 'Mexico'
    }
    df = df.copy()
    df['Country'] = df['Country'].replace(country_corrections)

    display_name_map = {
        'United States': 'USA',
        'United Arab Emirates': 'UAE',
        'South Korea': 'Korea',
        'Mexico': 'Mexico',
    }

    # --- 标准化 force_inside_countries 中的国家名 ---
    force_inside_countries = [country_corrections.get(c, c) for c in force_inside_countries]

    required_cols = {indicator, width_indicator, 'gdp'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for plot_variable_width_bar: {missing_cols}")

    df = df.dropna(subset=[indicator, width_indicator, 'gdp'])

    # --- 从CSV文件加载全球北方国家列表 ---
    global_north_countries = set()
    if global_north_file is None:
        print("Warning: No global_north_file provided. Assuming all countries are Global South.")
    else:
        import pandas as pd
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'gbk', 'iso-8859-1']
        df_north = None
        for enc in encodings_to_try:
            try:
                df_north = pd.read_csv(global_north_file, encoding=enc)
                print(f"Successfully read Global North list with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Failed to read with encoding {enc}: {e}")
                continue

        if df_north is not None and not df_north.empty:
            first_col = df_north.columns[0]
            country_list = df_north[first_col].dropna().astype(str).str.strip().tolist()
            global_north_countries = set(country_list)
            print(f"Loaded {len(global_north_countries)} countries from Global North list.")
        else:
            print(f"Error: Could not read Global North file '{global_north_file}' with any common encoding.")

    # --- 聚合：City -> Country ---
    agg_dict = {width_indicator: 'sum', indicator: 'mean', 'gdp': 'mean'}
    df_country = df.groupby('Country').agg(agg_dict).reset_index()

    df_country['is_global_north'] = df_country['Country'].isin(global_north_countries)
    df_country['is_global_south'] = ~df_country['is_global_north']

    # Sort by indicator (Y-axis) from low to high to create a staircase effect
    df_country = df_country.sort_values(by=indicator, ascending=True)

    # Calculate width (X-axis) based on the specified width_indicator
    total_width_val = df_country[width_indicator].sum()
    if total_width_val == 0:
        print(f"Warning: Total sum of '{width_indicator}' is zero. Cannot create width-based bars.")
        return

    df_country['width'] = (df_country[width_indicator] / total_width_val) * 100
    df_country['left'] = df_country['width'].cumsum().shift(1).fillna(0)

    print(f"Plotting Variable Width Bar: Indicator='{indicator}', Width='{width_indicator}'")
    print(f"Countries: {len(df_country)}, Total {width_indicator} Share: {df_country['width'].sum():.1f}%")

    # --- 保存国家统计 CSV ---
    if save_path and 'pop_size' in df_country.columns:
        try:
            output_dir = os.path.dirname(save_path)
            csv_path = f'{output_dir}/deployment_compatibility_{method}_country_stats.csv'

            # 包含 is_global_north 列（转为 0/1）
            df_csv = df_country[['Country', indicator, 'pop_size', 'is_global_north']].copy()

            total_pop = df_csv['pop_size'].sum()
            if total_pop > 0:
                df_csv['pop_share_%'] = (df_csv['pop_size'] / total_pop) * 100
            else:
                df_csv['pop_share_%'] = 0

            # 重命名指标列
            df_csv = df_csv.rename(columns={indicator: 'avg_deployment_compatibility_gap'})

            # 将布尔值转为整数 0/1
            df_csv['is_global_north'] = df_csv['is_global_north'].astype(int)

            df_csv.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Save to: {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to save country stats CSV: {e}")

    # --- 计算统计值 ---
    global_avg = df_country[indicator].mean()
    global_north_avg = df_country[df_country['is_global_north']][indicator].mean() if df_country['is_global_north'].any() else np.nan
    global_south_avg = df_country[df_country['is_global_south']][indicator].mean() if df_country['is_global_south'].any() else np.nan

    # Create the figure and axis
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes([0.10, 0.15, 0.85, 0.75])

    # --- Plot Variable Width Bars ---
    norm = mcolors.Normalize(vmin=df_country['gdp'].min(), vmax=df_country['gdp'].max())
    cmap = cm.get_cmap('viridis')
    bar_colors = cmap(norm(df_country['gdp']))

    bars = ax.bar(
        x=df_country['left'],
        height=df_country[indicator],
        width=df_country['width'],
        align='edge',
        color=bar_colors,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.85
    )

    # --- 添加国家标签 ---
    label_threshold = 3.0  # 宽度阈值，低于此值的国家名用箭头指向

    for idx, row in df_country.iterrows():
        country_display = display_name_map.get(row['Country'], row['Country'])  # 使用显示名称映射

        if row['width'] < min_width_threshold:
            continue  # 不显示标签

        bar_center_x = row['left'] + row['width'] / 2
        bar_top_y = row[indicator]

        # 如果国家在 force_inside_countries 中，强制放内部
        if country_display in force_inside_countries:
            ax.text(
                x=bar_center_x,
                y=bar_top_y / 2,  # 放在条的垂直中心
                s=country_display,
                ha='center',
                va='center',
                fontsize=14,
                color='white',
                zorder=11
            )
            continue  # 跳过后续箭头逻辑

        if row['width'] > label_threshold:
            # 宽条：标签放在条内部
            ax.text(
                x=bar_center_x,
                y=bar_top_y / 2,  # 放在条的垂直中心
                s=country_display,
                ha='center',
                va='center',
                fontsize=14,
                color='white',
                zorder=11
            )
        else:
            # 窄条：标签放在条上方
            label_y = bar_top_y + (df_country[indicator].max() * 0.1)  # 增加偏移量 → 箭头更长
            ax.annotate(
                country_display,
                xy=(bar_center_x, bar_top_y),  # 指向条的顶部中心
                xytext=(bar_center_x, label_y),  # 标签位置（更高）
                ha='center',
                va='bottom',
                fontsize=14,
                color='black',
                zorder=11,
                arrowprops=dict(
                    arrowstyle='->',
                    color='black',
                    lw=2,         # 增加线宽
                    shrinkA=0,      # 箭头起点不缩进
                    shrinkB=0       # 箭头终点不缩进
                )
            )

    # --- 设置坐标轴范围 ---
    ax.set_xlim(0, 100)
    y_max = df_country[indicator].max()
    ax.set_ylim(0, y_max * 1.5)

    # --- 处理科学计数法 ---
    sci_indicators = ["gdp_sum", "pop_size"]
    if indicator in sci_indicators:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
        fig.canvas.draw()
        offset_str = ax.yaxis.get_offset_text().get_text()
        if offset_str:
            ax.yaxis.offsetText.set_visible(False)
            ax.text(0, 1.02, offset_str, transform=ax.transAxes, ha='left', va='bottom')

    # --- 添加全球南方、全球北方和全球平均线 ---
    if not np.isnan(global_south_avg):
        ax.axhline(y=global_south_avg, color='green', linestyle='--', linewidth=2, zorder=5, label='Average in the Global South')
    ax.axhline(y=global_avg, color='blue', linestyle='-.', linewidth=2, zorder=5, label='Global average')
    if not np.isnan(global_north_avg):
        ax.axhline(y=global_north_avg, color='red', linestyle=':', linewidth=2, zorder=5, label='Average in the Global North')

    # --- 用红色三角形标记全球北方国家 ---
    north_countries_data = df_country[df_country['is_global_north']]
    if not north_countries_data.empty:
        for _, row in north_countries_data.iterrows():
            triangle_x = row['left'] + row['width'] / 2
            triangle_y = row[indicator]
            ax.scatter(
                triangle_x, triangle_y,
                marker='^',
                color='red',
                s=120,
                zorder=15,
                edgecolor='black',
                linewidth=1
            )

    # --- 设置左上角图例 ---
    south_label = f'Average in the Global South = {global_south_avg:.2f}' if not np.isnan(global_south_avg) else 'Average in the Global South'
    global_label = f'Global average = {global_avg:.2f}'
    north_label = f'Average in the Global North = {global_north_avg:.2f}' if not np.isnan(global_north_avg) else 'Average in the Global North'

    legend_elements = [
        mlines.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=south_label),
        mlines.Line2D([0], [0], color='blue', linestyle='-.', linewidth=2, label=global_label),
        mlines.Line2D([0], [0], color='red', linestyle=':', linewidth=2, label=north_label),
        mlines.Line2D([0], [0], marker='^', color='red', markersize=16, label='Global North countries', linestyle='None')
    ]
    # Updated legend with white border/frame and larger font
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(0, 1), frameon=True,
              facecolor='white', edgecolor='gray',
              framealpha=1, fontsize=20)

    # --- 右上角添加人均GDP图例 ---
    cax = fig.add_axes([0.67, 0.85, 0.25, 0.03])

    # 创建 ScalarMappable 对象用于 colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=cax, orientation='horizontal')

    # 设定刻度定位器：每隔 5000 一个刻度
    cb.locator = ticker.MaxNLocator(nbins=5)

    cb.formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x / 10000)}')

    cb.update_ticks()
    cb.ax.tick_params(labelsize=20)

    # 设置颜色条标签
    cb.set_label(r'GDP per capita ($10^4$ US\$)', labelpad=8, fontsize=20)

    # --- 设置坐标轴标签 ---
    if width_indicator == 'gdp_sum':
        ax.set_xlabel(f"Cumulative share of GDP (USD) (%)", fontsize=20)
    elif width_indicator == 'pop_size':
        ax.set_xlabel(f"Cumulative share of the population size (%)", fontsize=20)
    else:
        ax.set_xlabel(f"Cumulative share of {width_indicator} (%)", fontsize=20)
    ax.set_ylabel("Deployment compatibility gap", fontsize=20)

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Save to: {save_path}")

    plt.close()


if __name__ == '__main__':
    source_directory = f'src/Results/Global_Comp/{THRESHOLD}/'
    results_directory = f'src/Results/Bias_Analysis/{THRESHOLD}/'

    # 1. Plot Pairplot and Correlation
    target_metrics1 = ['gdp', 'pop_density', 'annual_prep', 'snowfall', 'max_temp', 'min_temp',
                       'slope', 'road_density', 'sinuosity', 'ratio_complex', 'poi_richness']
    df_combined = load_and_process_data(target_metrics1, source_directory)
    plot_correlation(df_combined, target_metrics1, results_directory)

    # 2. Plot Metric vs Metric
    fixed_combinations = [
        ("gdp", "pop_density"),
        ("annual_prep", "snowfall"),
        ("slope", "sinuosity")
    ]

    for metric1, metric2 in fixed_combinations:
        df_combined = load_and_process_data([metric1, metric2], source_directory)
        save_path = f"{results_directory}{metric1}_{metric2}.png"
        plot_metric_vs_metric(
            combined_df=df_combined,
            metric1=metric1,
            metric2=metric2,
            save_path=save_path
        )

    # 3. Plot Radar Chart
    metric_directions = {
        'gdp': True,            # 高GDP = 外圈（展示经济实力）
        'pop_density': True,       # 高人口 = 外圈
        'annual_prep': True,    # 高降雨 = 外圈（展示降雨量）
        'snowfall': True,       # 高降雪 = 外圈
        'max_temp': True,       # 高温 = 外圈
        'min_temp': True,       # 数值大（暖和）= 外圈；数值小（冷）= 内圈。
        'slope': True,          # 高坡度 = 外圈
        'entropy': True,  # 高熵 = 外圈
        'road_density': True,   # 高坡度 = 外圈
        'sinuosity': True, # 高曲率 = 外圈
        'ratio_complex': True,  # 高复杂度 = 外圈
        'poi_richness': True,    # 高POI = 外圈
    }
    target_metrics3 = ['gdp', 'pop_density', 'annual_prep', 'snowfall', 'max_temp', 'min_temp',
                       'slope', 'road_density', 'sinuosity', 'ratio_complex', 'poi_richness']
    df_combined = load_and_process_data(target_metrics3, source_directory)
    plot_radar_chart(df_combined, target_metrics3, results_directory+'radar_chart.png', metric_directions)

    # 4. Plot the global heatmap and related analyses
    metric_directions = {
        'gdp': 1,  # 钱越多越好
        'pop_density': 1,  # 人越多越好
        'pop_size': 1,  # 人越多越好
        'annual_prep': -1,  # 雨越少越好
        'extreme_prep': -1,  # 暴雨越少越好
        'snowfall': -1,  # 雪越少越好
        'max_temp': -1,  # 极热不好，越低越好
        'min_temp': -1,  # 极寒不好，越低越好
        'slope': -1,  # 坡度越小越好
        'entropy': -1,  # 道路越乱(熵越高)越不好
        'road_density': 1,  # 路越多越好
        "sinuosity": -1,  # 道路越弯曲越不好
        'ratio_complex': -1,  # 复杂路口越少越好
        'poi_richness': -1,  # POI越多通常代表需求越杂
        'poi_entropy': -1  # POI越多通常代表需求越杂
    }

    metric_groups = {
        "socioeconomic": ["gdp", "pop_density"],
        "climatic": ["annual_prep", "extreme_prep", "snowfall", "max_temp", "min_temp"],
        "infrastructural": ["slope", "road_density", "sinuosity", "ratio_complex", "entropy"],
        "functional": ["poi_richness"],
    }

    # Define the lists to iterate over
    robustness_options = [True, False] # using only satellite-derived indicators
    method_options = ['topsis', 'vikor'] # multi-criteria decision-making method

    # Loop through robustness checks
    for robustness_check in robustness_options:
        # Loop through methods
        for method in method_options:

            print(f"\n\033[1;36mRunning analysis with robustness_check={robustness_check} and method={method}\033[0m\n")

            # Define metrics based on robustness_check
            if robustness_check:
                target_metrics4 = ['gdp', 'pop_density', 'annual_prep', 'snowfall', 'max_temp', 'min_temp',
                                   'slope']
            else:
                target_metrics4 = ['gdp', 'pop_density', 'annual_prep', 'snowfall', 'max_temp', 'min_temp',
                                   'slope', 'road_density', 'sinuosity', 'ratio_complex', 'poi_richness']

            # Load data with the extended list
            df_combined = load_and_process_data(target_metrics4 + ['gdp_sum', 'pop_size'], source_directory)

            # Run the deployment gap analysis
            df_final = plot_deployment_gap(df_combined, target_metrics4,
                                           save_path=results_directory + method + '/' + str(robustness_check),
                                           method=method, metric_directions=metric_directions, weight_method='entropy')

            plot_gap_drivers(df_final, target_metrics4,
                             save_path=results_directory + method + '/' + str(robustness_check) + '/Drivers')

            plot_regional_inequality(df_final, save_path=results_directory + method + '/' + str(
                robustness_check) + '/regional_inequality.png')

            plot_gap_contribution(df_final, target_metrics4, metric_directions,
                save_path=results_directory + method + '/' + str(robustness_check) + '/gap_contribution_top10.png',
                top_n=10
            )

            plot_top_potential_cities(df_final, save_path=results_directory + method + '/' + str(
                robustness_check) + '/potential_cities.png')

            plot_gap_cdf(df_final, save_path=results_directory + method + '/' + str(robustness_check) + '/gap_cdf.png')

            plot_logistic_curve(df_final, save_path=results_directory + method + '/' + str(
                robustness_check) + '/logistic_curve.png')

            # Use 'gdp_sum' or 'pop_size' for the width of the bars
            if 'gdp_sum' in df_combined.columns:
                plot_variable_width_bar(df_final, method, save_path=results_directory + method + '/' + str(
                    robustness_check) + '/country_inequality_gdp_sum.png',
                                        indicator="deployment_compatibility_gap", width_indicator='gdp_sum')

            if 'pop_size' in df_combined.columns:
                plot_variable_width_bar(df_final, method, save_path=results_directory + method + '/' + str(
                    robustness_check) + '/country_inequality_pop_size.png',
                                        indicator="deployment_compatibility_gap", width_indicator='pop_size')