import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import scienceplots
import warnings
warnings.filterwarnings('ignore')
from utilities.hyperparameters import THRESHOLD, LOG_TRANSFORM_VARIABLES, LABELS, LABELS_UNITLESS


# Hyperparameters of scienceplots
plt.style.use(['science', 'no-latex', 'nature'])

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
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

class EntropyBalancingAnalysis:
    def __init__(self, data_path, covariates, target_variables, log_transform_variables, result_dir='Results'):
        self.data_path = data_path
        self.covariates = covariates
        self.target_variables = target_variables
        self.log_transform_variables = log_transform_variables
        self.result_dir = result_dir
        self.df = None
        self.matched_df = None
        self.balance_results = None
        self.reg_results_list = []
        self.balance_stats = []
        self.success_countries = []
        self.country_smd_results = []
        self.within_country_results = pd.DataFrame()

        # Create result directory
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            print(f"Creating result directory: {self.result_dir}")

    # ==========================================
    # 1. Core Algorithms & Statistical Functions
    # ==========================================
    @staticmethod
    def entropy_balancing(X_control, target_moments, max_iter=10000, tol=1e-6):
        n_control, n_features = X_control.shape
        if np.any(np.isnan(X_control)) or np.any(np.isnan(target_moments)):
            raise ValueError("Input data contains NaN values")

        X_mean = np.mean(X_control, axis=0)
        X_centered = X_control - X_mean
        target_centered = target_moments - X_mean

        def loss_func(zeta):
            linear_pred = np.dot(X_centered, zeta)
            max_val = np.max(linear_pred)
            log_sum_exp = max_val + np.log(np.sum(np.exp(linear_pred - max_val)))
            return log_sum_exp - np.dot(target_centered, zeta)

        def grad_func(zeta):
            linear_pred = np.dot(X_centered, zeta)
            max_val = np.max(linear_pred)
            weights = np.exp(linear_pred - max_val)
            weights = weights / np.sum(weights)
            weighted_moments = np.dot(weights, X_centered)
            return weighted_moments - target_centered

        zeta_init = np.zeros(n_features)

        result = minimize(loss_func, zeta_init, method='L-BFGS-B', jac=grad_func,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})

        linear_pred = np.dot(X_centered, result.x)
        max_val = np.max(linear_pred)
        weights = np.exp(linear_pred - max_val)
        normalized_weights = weights / np.sum(weights)

        if not np.all(np.isfinite(normalized_weights)):
            return np.full_like(weights, np.nan), False

        return normalized_weights, result.success

    @staticmethod
    def weighted_mean(x, w):
        x, w = np.array(x), np.array(w)
        valid_mask = ~np.isnan(x) & ~np.isnan(w)
        if np.sum(w[valid_mask]) == 0: return np.nan
        return np.sum(x[valid_mask] * w[valid_mask]) / np.sum(w[valid_mask])

    @staticmethod
    def weighted_std(x, w, ddof=1):
        x, w = np.array(x), np.array(w)
        valid_mask = ~np.isnan(x) & ~np.isnan(w) & (w > 0)
        if np.sum(valid_mask) < 2: return np.nan
        x_valid, w_valid = x[valid_mask], w[valid_mask]
        w_sum = np.sum(w_valid)
        if w_sum == 0: return np.nan
        mean_val = np.sum(x_valid * w_valid) / w_sum
        weighted_var = np.sum(w_valid * (x_valid - mean_val) ** 2) / w_sum
        weighted_var = max(0, weighted_var)
        if ddof > 0:
            ess = (w_sum ** 2) / np.sum(w_valid ** 2)
            if ess > ddof + 1e-10:
                weighted_var = weighted_var * ess / (ess - ddof)
        return np.sqrt(weighted_var)

    @staticmethod
    def weighted_ks_statistic(data1, weights1, data2, weights2):
        """
        Compute Weighted Kolmogorov-Smirnov Statistic.
        Returns the max distance between two weighted CDFs.
        """
        # Sort data and reorder weights
        ind1 = np.argsort(data1)
        d1 = data1[ind1]
        w1 = weights1[ind1]

        ind2 = np.argsort(data2)
        d2 = data2[ind2]
        w2 = weights2[ind2]

        # Compute empirical CDFs
        cdf1 = np.cumsum(w1) / np.sum(w1)
        cdf2 = np.cumsum(w2) / np.sum(w2)

        # To compare, we need to evaluate them at the same points.
        # We use all unique data points from both sets as the evaluation grid.
        all_points = np.sort(np.concatenate([d1, d2]))

        # Interpolate CDFs to the common grid
        # use 'right' to respect the definition of CDF (P(X <= x))
        cdf1_interp = np.interp(all_points, d1, cdf1, left=0, right=1)
        cdf2_interp = np.interp(all_points, d2, cdf2, left=0, right=1)

        # KS statistic is the maximum absolute difference
        ks_stat = np.max(np.abs(cdf1_interp - cdf2_interp))
        return ks_stat

    # ==========================================
    # 2. Data Loading & Preprocessing
    # ==========================================
    def load_and_preprocess(self):
        print("=" * 60)
        print("1. Data Loading and Preprocessing")
        print("=" * 60)
        np.random.seed(42)

        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
        except FileNotFoundError:
            print("Error: File not found, please check the path")
            raise

        rename_map = {'USA': 'United States', 'UAE': 'United Arab Emirates', 'Korea': 'South Korea'}
        if 'Country' in self.df.columns:
            self.df['Country'] = self.df['Country'].replace(rename_map)

        if 'group' in self.df.columns:
            self.df['treatment'] = (self.df['group'] == 'AV-served').astype(int)
        elif 'treatment' not in self.df.columns:
            raise ValueError("Dataset must contain 'group' or 'treatment' column.")

        self.df['eb_weight'] = np.where(self.df['treatment'] == 1, 1.0, np.nan)

        all_numeric_cols = self.covariates + self.target_variables
        for col in all_numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        log_cols_to_replace = []
        for col in self.log_transform_variables:
            if col in self.df.columns:
                if (self.df[col] <= -1).any():
                    self.df.loc[self.df[col] <= -1, col] = np.nan
                log_col = f"{col}_log"
                self.df[log_col] = np.log1p(self.df[col])
                log_cols_to_replace.append(col)

        self.df.drop(columns=log_cols_to_replace, inplace=True)

        def replace_with_log(names, log_vars):
            return [f"{name}_log" if name in log_vars else name for name in names]

        self.covariates = replace_with_log(self.covariates, self.log_transform_variables)
        self.target_variables = replace_with_log(self.target_variables, self.log_transform_variables)

        self.df = self.df.dropna(subset=self.covariates + ['Country', 'treatment'])

    # ==========================================
    # 3. Execute Entropy Balancing
    # ==========================================
    def run_entropy_balancing(self):
        countries = sorted(self.df['Country'].unique())
        print(f"Number of countries to process: {len(countries)}")

        for i, country in enumerate(countries, 1):
            country_mask = self.df['Country'] == country
            country_df = self.df[country_mask]
            treated_mask = country_df['treatment'] == 1
            control_mask = country_df['treatment'] == 0
            n_treated, n_control = treated_mask.sum(), control_mask.sum()

            if n_treated < 2 or n_control < 2:
                continue

            X_treated = country_df.loc[treated_mask, self.covariates].values
            X_control = country_df.loc[control_mask, self.covariates].values

            if np.any(np.isnan(X_treated)) or np.any(np.isnan(X_control)):
                continue

            try:
                target_moments = np.mean(X_treated, axis=0)
                weights, success = self.entropy_balancing(X_control, target_moments)

                if not success:
                    continue

                final_weights = weights * n_treated
                if np.any(final_weights < 0):
                    continue

                self.df.loc[country_mask & control_mask, 'eb_weight'] = final_weights
                ess = (np.sum(final_weights) ** 2) / np.sum(final_weights ** 2)
                self.success_countries.append(country)
                self.balance_stats.append(
                    {'Country': country, 'Status': 'Success', 'N_treated': n_treated, 'N_control': n_control,
                     'ESS': ess})
                print(f"  [{i:2d}/{len(countries)}] {country:<20}: Success (ESS={ess:5.1f})")

            except Exception as e:
                print(f"  [{i:2d}/{len(countries)}] {country:<20}: Error -> {str(e)[:50]}...")

        self.matched_df = self.df.dropna(subset=['eb_weight']).copy()
        print(f"\n✅ Entropy Balancing Completed! Matched samples: {len(self.matched_df)}")

    # ==========================================
    # 4. Balance Check
    # ==========================================
    def check_balance(self):
        print("\n" + "=" * 60 + "\n5. Covariate Balance Check\n" + "=" * 60)
        overall_results = []
        self.country_smd_results = []

        for country in self.success_countries:
            country_df = self.matched_df[self.matched_df['Country'] == country]
            treated = country_df[country_df['treatment'] == 1]
            control = country_df[country_df['treatment'] == 0]

            for cov in self.covariates:
                mean_t_u, mean_c_u = treated[cov].mean(), control[cov].mean()
                sd_t_u, sd_c_u = treated[cov].std(ddof=1), control[cov].std(ddof=1)
                smd_u = abs(mean_t_u - mean_c_u) / np.sqrt((sd_t_u ** 2 + sd_c_u ** 2) / 2)

                mean_t_w = self.weighted_mean(treated[cov], treated['eb_weight'])
                mean_c_w = self.weighted_mean(control[cov], control['eb_weight'])
                var_t_w = self.weighted_std(treated[cov], treated['eb_weight'], ddof=1) ** 2
                var_c_w = self.weighted_std(control[cov], control['eb_weight'], ddof=1) ** 2

                if np.isnan(var_t_w) or np.isnan(var_c_w) or (var_t_w + var_c_w) == 0:
                    smd_w = np.nan
                else:
                    smd_w = abs(mean_t_w - mean_c_w) / np.sqrt((var_t_w + var_c_w) / 2)

                self.country_smd_results.append({
                    'Country': country,
                    'Covariate': cov,
                    'Weighted_SMD': smd_w
                })

                if country == self.success_countries[0]:
                    overall_results.append({
                        'Covariate': cov, 'Unweighted_SMD': smd_u, 'Weighted_SMD': smd_w,
                        'Is_Balanced': (smd_w < 0.1 if not np.isnan(smd_w) else False)
                    })

        self.balance_results = pd.DataFrame(overall_results)
        print(self.balance_results.to_string(index=False, float_format="%.4f"))

    # ==========================================
    # 5. Within-Country Regression Analysis
    # ==========================================
    def run_within_country_regressions(self):
        print("\n" + "=" * 60)
        print("6. Within-Country Regression Analysis (WLS)")
        print("=" * 60)
        results = []
        for country in self.success_countries:
            country_df = self.matched_df[self.matched_df['Country'] == country].copy()
            if len(country_df) < 10: continue

            control_weights = country_df[country_df['treatment'] == 0]['eb_weight']
            if len(control_weights) > 0 and control_weights.mean() > 0:
                if control_weights.max() / control_weights.mean() > 100:
                    continue

            for target in self.target_variables:
                if target not in country_df.columns: continue
                temp_df = country_df.dropna(subset=[target]).copy()
                if len(temp_df) < 5 or temp_df['treatment'].nunique() < 2: continue

                Y = temp_df[target]
                X = sm.add_constant(temp_df['treatment'])
                weights = temp_df['eb_weight']

                try:
                    model = sm.WLS(Y, X, weights=weights).fit(cov_type='HC3')
                    results.append({
                        'Country': country,
                        'Variable': target,
                        'Coefficient': model.params['treatment'],
                        'Std_Error': model.bse['treatment'],
                        'P_Value': model.pvalues['treatment'],
                        'CI_Lower': model.conf_int().loc['treatment'][0],
                        'CI_Upper': model.conf_int().loc['treatment'][1],
                        'Significant_10': model.pvalues['treatment'] < 0.10,
                        'Significant_05': model.pvalues['treatment'] < 0.05
                    })
                except Exception:
                    continue

        self.within_country_results = pd.DataFrame(results)
        print(f"✅ Regressions completed. Found {len(self.within_country_results)} results.")

    # ==========================================
    # ROBUSTNESS CHECKS
    # ==========================================
    def run_all_robustness_checks(self):
        """Master function to run all robustness checks"""
        if self.within_country_results.empty:
            print("Skipping robustness checks (no regression results found).")
            return

        print("\n" + "=" * 60 + "\n8. Robustness Checks\n" + "=" * 60)
        self.run_robustness_weight_trimming()
        self.run_robustness_permutation()
        self.run_robustness_nonparametric()

    def run_robustness_weight_trimming(self, trim_quantile=0.95):
        """
        Robustness Check 1: Weight Trimming
        Caps weights at the 95th percentile to ensure results aren't driven by extreme outliers.
        """
        print(f"\n[Robustness 1] Weight Trimming (Cap at {trim_quantile * 100:.0f}th percentile)...")
        results = []

        # Iterate only over significant results to save time
        sig_results = self.within_country_results[self.within_country_results['Significant_10']]

        for _, row in sig_results.iterrows():
            country = row['Country']
            target = row['Variable']
            original_coef = row['Coefficient']

            country_df = self.matched_df[self.matched_df['Country'] == country].copy()
            temp_df = country_df.dropna(subset=[target]).copy()

            # Trim weights
            weights = temp_df['eb_weight'].values
            cap_val = np.percentile(weights, trim_quantile * 100)
            trimmed_weights = np.minimum(weights, cap_val)

            try:
                Y = temp_df[target]
                X = sm.add_constant(temp_df['treatment'])
                model = sm.WLS(Y, X, weights=trimmed_weights).fit(cov_type='HC3')
                new_coef = model.params['treatment']
                new_p = model.pvalues['treatment']

                results.append({
                    'Country': country,
                    'Variable': target,
                    'Original_Coef': original_coef,
                    'Trimmed_Coef': new_coef,
                    'Trimmed_P_Value': new_p,
                    'Coef_Change_Pct': (new_coef - original_coef) / original_coef * 100,
                    'Consistent_Sign': np.sign(new_coef) == np.sign(original_coef)
                })
            except:
                continue

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            save_path = os.path.join(self.result_dir, 'robustness_trimming.csv')
            res_df.to_csv(save_path, index=False)
            consistent_count = res_df['Consistent_Sign'].sum()
            print(f"  Saved results to {save_path}")
            print(
                f"  Consistency: {consistent_count}/{len(res_df)} ({consistent_count / len(res_df) * 100:.1f}%) retained sign.")
        else:
            print("  No significant results to check.")

    def run_robustness_permutation(self, n_permutations=5000):
        """
        Robustness Check 2: Permutation Test (Placebo Test)
        Shuffles the OUTCOME variable (Y) to break the relationship with treatment,
        while keeping the covariate balance structure (weights) intact.
        """
        print(f"\n[Robustness 2] Permutation Test (Shuffling Y, N={n_permutations})...")
        results = []

        # Check only significant results
        sig_results = self.within_country_results[self.within_country_results['Significant_10']]

        for _, row in sig_results.iterrows():
            country = row['Country']
            target = row['Variable']
            real_coef = row['Coefficient']

            country_df = self.matched_df[self.matched_df['Country'] == country].copy()
            temp_df = country_df.dropna(subset=[target]).copy()

            if len(temp_df) < 10: continue

            perm_coefs = []
            weights = temp_df['eb_weight'].values
            X = sm.add_constant(temp_df['treatment'])
            original_Y = temp_df[target].values

            # Run permutations
            for _ in range(n_permutations):
                shuffled_Y = np.random.permutation(original_Y)

                try:
                    # Regress Shuffled Y on Fixed X with Fixed Weights
                    model = sm.WLS(shuffled_Y, X, weights=weights).fit(cov_type='HC3')
                    perm_coefs.append(model.params['treatment'])
                except:
                    pass

            perm_coefs = np.array(perm_coefs)

            if len(perm_coefs) > 0:
                # Calculate Empirical P-value (two-sided)
                # 检查真实系数是否落在随机分布的两端
                n_extreme = np.sum(np.abs(perm_coefs) >= np.abs(real_coef))
                emp_p_val = n_extreme / len(perm_coefs)

                results.append({
                    'Country': country,
                    'Variable': target,
                    'Real_Coef': real_coef,
                    'Perm_Mean': np.mean(perm_coefs),  # 理论上应该接近 0
                    'Perm_Std': np.std(perm_coefs),
                    'Empirical_P_Value': emp_p_val,
                    'Robust': emp_p_val < 0.1
                })

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            save_path = os.path.join(self.result_dir, 'robustness_permutation.csv')
            res_df.to_csv(save_path, index=False)
            robust_count = res_df['Robust'].sum()
            print(f"  Saved results to {save_path}")
            print(f"  Robustness: {robust_count}/{len(res_df)} passed permutation test (p<0.1).")

    def run_robustness_nonparametric(self):
        """
        Robustness Check 3: Weighted Non-parametric Test (KS Test)
        Compares the entire distribution shape, not just the mean.
        """
        print(f"\n[Robustness 3] Weighted Kolmogorov-Smirnov Test...")
        results = []

        sig_results = self.within_country_results[self.within_country_results['Significant_10']]

        for _, row in sig_results.iterrows():
            country = row['Country']
            target = row['Variable']

            country_df = self.matched_df[self.matched_df['Country'] == country].copy()
            temp_df = country_df.dropna(subset=[target]).copy()

            treated = temp_df[temp_df['treatment'] == 1]
            control = temp_df[temp_df['treatment'] == 0]

            if len(treated) < 5 or len(control) < 5: continue

            # Weighted KS Statistic
            ks_stat = self.weighted_ks_statistic(
                treated[target].values, treated['eb_weight'].values,
                control[target].values, control['eb_weight'].values
            )

            # Approximate P-value for KS (using standard asymptotic formula, though weights make it approximate)
            n1 = treated['eb_weight'].sum()  # Effective N roughly
            n2 = control['eb_weight'].sum()
            m = (n1 * n2) / (n1 + n2)
            # Kolmogorov distribution approximation
            # P(D > z) approx 2 * exp(-2 * z^2 * m) ? No, standard formula:
            # D_alpha = c(alpha) * sqrt((n1+n2)/n1n2).
            # Let's use scipy's kstwo prob function with effective sample size
            # This is a heuristic for weighted data.
            try:
                p_val = stats.kstwo.sf(ks_stat, np.round(n1 + n2))  # Rough approx
            except:
                p_val = np.nan

            results.append({
                'Country': country,
                'Variable': target,
                'KS_Statistic': ks_stat,
                'Approx_P_Value': p_val,
                'Significant_Diff': ks_stat > 0.2  # Arbitrary threshold or use p-val
            })

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            save_path = os.path.join(self.result_dir, 'robustness_nonparametric_ks.csv')
            res_df.to_csv(save_path, index=False)
            print(f"  Saved results to {save_path}")
            print(f"  Mean KS Statistic: {res_df['KS_Statistic'].mean():.3f}")

    # ==========================================
    # 6. Visualization
    # ==========================================
    def plot_covariate_density(self):
        if not self.covariates:
            return

        # 动态计算行数，固定 2 列 (也可改为 3 列)
        n_cols = 2
        n_rows = (len(self.covariates) + n_cols - 1) // n_cols

        # 适当增加高度以容纳顶部图例
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        treated_df = self.matched_df[self.matched_df['treatment'] == 1]
        control_df = self.matched_df[self.matched_df['treatment'] == 0]

        # 定义颜色方案
        color_treated = '#D62728'  # Muted Red
        color_control_raw = '#7F7F7F'  # Gray
        color_control_w = '#1F77B4'  # Muted Blue

        for i, cov in enumerate(self.covariates):
            ax = axes[i]

            # 提取数据
            t_data = treated_df[cov].dropna().values
            c_data = control_df[cov].dropna().values
            c_weights = control_df['eb_weight'].loc[control_df[cov].notna()].values

            # --- 1. 绘制 Control Raw (灰色虚线，作为背景基准) ---
            if len(np.unique(c_data)) > 1:
                try:
                    density_c = gaussian_kde(c_data)
                    xs_c = np.linspace(c_data.min(), c_data.max(), 200)
                    ax.plot(xs_c, density_c(xs_c), color=color_control_raw,
                            linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
                except:
                    pass  # Fallback handled below if needed, usually KDE works

            # --- 2. 绘制 Treated (红色实线 + 填充) ---
            if len(np.unique(t_data)) > 1:
                try:
                    density_t = gaussian_kde(t_data)
                    xs_t = np.linspace(t_data.min(), t_data.max(), 200)
                    ax.plot(xs_t, density_t(xs_t), color=color_treated,
                            linewidth=2, zorder=3)
                    ax.fill_between(xs_t, density_t(xs_t), color=color_treated,
                                    alpha=0.2, zorder=3)
                except:
                    ax.hist(t_data, density=True, color=color_treated, alpha=0.3)

            # --- 3. 绘制 Control Weighted (蓝色实线 + 填充) ---
            if len(np.unique(c_data)) > 1 and len(c_weights) == len(c_data):
                try:
                    density_cw = gaussian_kde(c_data, weights=c_weights)
                    xs_cw = np.linspace(c_data.min(), c_data.max(), 200)
                    ax.plot(xs_cw, density_cw(xs_cw), color=color_control_w,
                            linewidth=2, linestyle='-', zorder=2)
                    ax.fill_between(xs_cw, density_cw(xs_cw), color=color_control_w,
                                    alpha=0.15, zorder=2)
                except:
                    pass

            # --- 4. 美化坐标轴 ---
            # 设置标签
            ax.set_xlabel(LABELS.get(cov, cov), labelpad=10)

            # 仅在第一列显示 Y 轴标签，减少杂乱，或者全部显示视需求而定
            # 这里为了对齐美观，建议全部保留，但可以简化
            ax.set_ylabel('Density')

            # 强制显示四边边框
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor('black')

            # 刻度设置：减少刻度数量，使其更清爽
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.tick_params(axis='both', which='major', width=1.5, length=5)

            # --- 5. 处理多余的子图 ---
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # --- 6. 添加全局图例 (Global Legend) ---
        # 创建自定义图例句柄
        legend_elements = [
            Line2D([0], [0], color=color_treated, lw=2, label='Treated'),
            Line2D([0], [0], color=color_control_raw, lw=1.5, linestyle='--', label='Control (Raw)'),
            Line2D([0], [0], color=color_control_w, lw=2, label='Control (Weighted)')
        ]

        # 将图例放置在整个 Figure 的顶部中心
        fig.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)

        plt.tight_layout()
        # 留出顶部空间给图例
        plt.subplots_adjust(top=0.92)

        save_path = os.path.join(self.result_dir, 'covariate_density.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_love_plot(self):
        """Plotting Love Plot with prettified y-axis labels"""
        if self.balance_results is None:
            return

        fig, ax = plt.subplots(figsize=(10, 6))  # auto height
        df_plot = self.balance_results.copy()
        df_plot['Covariate_Label'] = df_plot['Covariate'].map(LABELS_UNITLESS).fillna(df_plot['Covariate'])
        df_plot = df_plot.sort_values('Unweighted_SMD', ascending=True)

        ax.scatter(df_plot['Unweighted_SMD'], df_plot['Covariate_Label'], label='Original',
                   color='gray', alpha=0.7, s=120, marker='o', edgecolors='k', linewidth=0.5)
        ax.scatter(df_plot['Weighted_SMD'], df_plot['Covariate_Label'], label='Weighted',
                   color='#d62728', alpha=0.9, s=120, marker='D', edgecolors='k', linewidth=0.5)

        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Threshold (0.1)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Standardized mean difference (SMD)')
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'love_plot.png'), dpi=300)
        plt.close()

    def plot_faceted_forest_plot(self):
        if self.within_country_results.empty:
            return

        df = self.within_country_results.copy()
        df['Variable_Label'] = df['Variable'].map(LABELS_UNITLESS).fillna(df['Variable'])
        countries = sorted(df['Country'].unique())

        for country in countries:
            country_data = df[df['Country'] == country].copy()
            if country_data.empty:
                continue

            # Sort by coefficient for consistent ordering
            country_data = country_data.sort_values('Coefficient').reset_index(drop=True)

            # 1. 设置画布风格
            fig, ax = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(len(country_data))

            # 2. 定义颜色逻辑：Coefficient > 0 为红色(促进)，Coefficient < 0 为蓝色(抑制)
            colors = ['#d62728' if x > 0 else '#1f77b4' for x in country_data['Coefficient']]

            # 3. 绘制背景条纹 (Zebra Striping)
            for i in range(len(country_data)):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color='gray', alpha=0.1, zorder=0)

            # 4. 绘制基准线 (保持原代码逻辑，以0为界)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

            # 5. 绘制误差棒和点 (手动绘制以匹配参考风格)
            for i, (idx, row) in enumerate(country_data.iterrows()):
                # 绘制误差线 (CI)
                ax.plot([row['CI_Lower'], row['CI_Upper']], [i, i],
                        color=colors[i], linewidth=2, alpha=0.6, zorder=2)

                # 绘制误差线端点 (Caps)
                # 端点高度设为 0.1 (可视情况调整)
                ax.plot([row['CI_Lower'], row['CI_Lower']], [i - 0.1, i + 0.1],
                        color=colors[i], linewidth=1.5, alpha=0.6, zorder=2)
                ax.plot([row['CI_Upper'], row['CI_Upper']], [i - 0.1, i + 0.1],
                        color=colors[i], linewidth=1.5, alpha=0.6, zorder=2)

                # 绘制中心点 (Coefficient) - 带白色描边
                ax.scatter(row['Coefficient'], i, color=colors[i], s=100, zorder=3, edgecolors='white')

                # --- 6. 添加文字标注 ---
                p = row['P_Value']
                sig_text = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                coef_text = f"{row['Coefficient']:.2f}"

                # 1. 绘制 Coefficient 值
                t = ax.text(row['Coefficient'], i + 0.15, coef_text,
                            ha='center', va='bottom',
                            fontweight='bold', color=colors[i], fontsize=20)

                if sig_text:
                    # 2. 使用 annotate 基于 Coefficient 文本的位置进行偏移
                    ax.annotate(sig_text,
                                xy=(1, 0), xycoords=t,  # 锚点设在 Coefficient 值的右下角
                                xytext=(2, -7), textcoords='offset points',
                                ha='left', va='bottom',  # 星号左对齐
                                fontweight='bold', color=colors[i], fontsize=24)

            # 7. 轴设置
            ax.set_yticks(y_pos)
            ax.set_yticklabels(country_data['Variable_Label'], fontsize=24)
            ax.tick_params(axis='x', labelsize=20)
            ax.set_xlabel('WLS regression coefficient', fontsize=24)

            plt.tight_layout()

            # Safe filename generation
            safe_country = (
                country
                .replace(" ", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .replace(",", "")
            )

            # Ensure directory exists
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)

            filepath = os.path.join(self.result_dir, f'forest_{safe_country}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_weight_diagnostics(self):
        """Weight Diagnostics Visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # 1. Boxplot
        ax = axes[0, 0]
        weight_data = [self.matched_df[self.matched_df['treatment'] == t]['eb_weight'] for t in [0, 1]]
        bp = ax.boxplot(weight_data, labels=['Control', 'Treated'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_title('Weight distribution')

        # 2. Q-Q Plot (Control)
        ax = axes[0, 1]
        stats.probplot(self.matched_df[self.matched_df['treatment'] == 0]['eb_weight'], dist="norm", plot=ax)
        ax.set_title('Control weights Q-Q plot')

        # 3. CDF
        ax = axes[0, 2]
        for t, label, color in [(0, 'Control', 'blue'), (1, 'Treated', 'red')]:
            w = np.sort(self.matched_df[self.matched_df['treatment'] == t]['eb_weight'])
            ax.plot(w, np.arange(1, len(w) + 1) / len(w), label=label, color=color)
        ax.set_title('Weight CDF')
        ax.legend()

        # 4. Weights vs Covariate
        ax = axes[1, 0]
        if self.covariates:
            cov = self.covariates[0]
            sc = ax.scatter(self.matched_df[cov], self.matched_df['eb_weight'],
                            c=self.matched_df['treatment'], cmap='coolwarm', alpha=0.6)
            ax.set_xlabel(LABELS.get(cov, cov))
            ax.set_ylabel('Weights')
            ax.set_title(f'Weights vs {cov}')
        else:
            ax.axis('off')

        # 5. ESS
        ax = axes[1, 1]
        ess_list = []
        n_list = []
        for t in [0, 1]:
            w = self.matched_df[self.matched_df['treatment'] == t]['eb_weight']
            ess_list.append((w.sum() ** 2) / (w ** 2).sum())
            n_list.append(len(w))

        x_pos = np.arange(2)
        ax.bar(x_pos - 0.2, n_list, 0.4, label='Actual N', alpha=0.5)
        ax.bar(x_pos + 0.2, ess_list, 0.4, label='ESS', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Control', 'Treated'])
        ax.set_title('Effective Sample Size')
        ax.legend()

        # 6. Extreme Weights
        ax = axes[1, 2]
        cw = self.matched_df[self.matched_df['treatment'] == 0]['eb_weight']
        ratio = cw / cw.mean()
        ax.hist(ratio, bins=30, alpha=0.7, color='steelblue')
        ax.axvline(x=10, color='red', linestyle='--')
        ax.set_xlabel('Weight / Mean ratio')
        ax.set_title('Extreme weight detection')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'weight_diagnostics.png'), dpi=300)
        plt.close()

    def plot_country_smd_heatmap(self):
        """Plot heatmap of Weighted SMD by Country and Covariate"""
        if not self.country_smd_results:
            return

        df_smd = pd.DataFrame(self.country_smd_results)
        heatmap_data = df_smd.pivot(index='Country', columns='Covariate', values='Weighted_SMD')

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)

        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index)
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_xticklabels([LABELS.get(c, c) for c in heatmap_data.columns], rotation=90, ha='right')

        # Annotate values
        for i in range(len(heatmap_data)):
            for j in range(len(heatmap_data.columns)):
                val = heatmap_data.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.25 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

        plt.colorbar(im, shrink=0.9, label='Weighted SMD')
        plt.title("Covariate Balance by Country")
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'country_smd_heatmap.png'), dpi=300)
        plt.close()

        print("✅ Saved country SMD heatmap.")

    def plot_overall_forest_plot(self):
        if self.within_country_results.empty:
            return

        df = self.within_country_results.copy()
        # Ensure LABELS is available
        df['Var_Label'] = df['Variable'].map(LABELS).fillna(df['Variable'])

        overall_results = []
        for var in df['Variable'].unique():
            var_df = df[df['Variable'] == var]
            if len(var_df) < 2:
                continue

            # Inverse variance weighting (Fixed Effects Model logic)
            weights = 1 / (var_df['Std_Error'] ** 2)
            coef_overall = np.sum(weights * var_df['Coefficient']) / np.sum(weights)
            se_overall = np.sqrt(1 / np.sum(weights))
            p_val = 2 * (1 - stats.norm.cdf(abs(coef_overall) / se_overall))
            ci_low, ci_high = coef_overall - 1.96 * se_overall, coef_overall + 1.96 * se_overall

            overall_results.append({
                'Variable': var,
                'Var_Label': LABELS_UNITLESS.get(var, var),
                'Coefficient': coef_overall,
                'Std_Error': se_overall,
                'P_Value': p_val,
                'CI_Lower': ci_low,
                'CI_Upper': ci_high
            })

        if not overall_results:
            return

        overall_df = pd.DataFrame(overall_results).sort_values('Coefficient').reset_index(drop=True)

        # 1. 设置画布风格
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(overall_df))

        # 2. 定义颜色逻辑：Coefficient > 0 为红色(促进)，Coefficient < 0 为蓝色(抑制)
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in overall_df['Coefficient']]

        # 3. 绘制背景条纹 (Zebra Striping)
        for i in range(len(overall_df)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color='gray', alpha=0.1, zorder=0)

        # 4. 绘制基准线 (以0为界)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

        # 5. 绘制误差棒和点
        for i, (idx, row) in enumerate(overall_df.iterrows()):
            # 绘制误差线 (CI)
            ax.plot([row['CI_Lower'], row['CI_Upper']], [i, i],
                    color=colors[i], linewidth=2, alpha=0.6, zorder=2)

            # 绘制误差线端点 (Caps)
            ax.plot([row['CI_Lower'], row['CI_Lower']], [i - 0.1, i + 0.1],
                    color=colors[i], linewidth=1.5, alpha=0.6, zorder=2)
            ax.plot([row['CI_Upper'], row['CI_Upper']], [i - 0.1, i + 0.1],
                    color=colors[i], linewidth=1.5, alpha=0.6, zorder=2)

            # 绘制中心点 (Coefficient) - 带白色描边
            ax.scatter(row['Coefficient'], i, color=colors[i], s=100, zorder=3, edgecolors='white')

            # --- 6. 添加文字标注 ---
            p = row['P_Value']
            sig_text = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            coef_text = f"{row['Coefficient']:.2f}"

            # 1. 绘制 Coefficient 值
            t = ax.text(row['Coefficient'], i + 0.15, coef_text,
                        ha='center', va='bottom',
                        fontweight='bold', color=colors[i], fontsize=20)

            if sig_text:
                # 2. 使用 annotate 基于 Coefficient 文本的位置进行偏移
                ax.annotate(sig_text,
                            xy=(1, 0), xycoords=t,  # 锚点设在 Coefficient 值的右下角
                            xytext=(2, -7), textcoords='offset points',
                            ha='left', va='bottom',  # 星号左对齐
                            fontweight='bold', color=colors[i], fontsize=24)

        # 7. 轴设置
        ax.set_yticks(y_pos)
        ax.set_yticklabels(overall_df['Var_Label'], fontsize=24)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_xlabel('WLS regression coefficient', fontsize=24)

        plt.tight_layout()

        plt.savefig(os.path.join(self.result_dir, 'overall_forest.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ==========================================
    # 7. Summary & Saving
    # ==========================================
    def export_country_tables(self, target_countries):
        """
        Export detailed tables for specific countries combining regression results
        and descriptive statistics (Weighted Means).
        """
        print("\n" + "=" * 60 + "\n9. Exporting Country-Specific Tables\n" + "=" * 60)

        if self.within_country_results.empty:
            print("No regression results available.")
            return

        for country in target_countries:
            # 1. 筛选该国家的回归结果
            reg_df = self.within_country_results[self.within_country_results['Country'] == country].copy()
            if reg_df.empty:
                print(f"No regression results found for {country}")
                continue

            # 2. 获取该国家的原始匹配数据
            match_df = self.matched_df[self.matched_df['Country'] == country].copy()

            # 3. 计算描述性统计 (Treated Mean & Weighted Control Mean)
            desc_stats = []
            for var in reg_df['Variable'].unique():
                # 提取该变量的非空数据
                temp_df = match_df.dropna(subset=[var])

                treated = temp_df[temp_df['treatment'] == 1]
                control = temp_df[temp_df['treatment'] == 0]

                if len(treated) == 0 or len(control) == 0:
                    continue

                # 计算均值
                mean_treated = treated[var].mean()
                # 注意：Control组必须用 eb_weight 计算加权均值
                mean_control_weighted = self.weighted_mean(control[var], control['eb_weight'])

                desc_stats.append({
                    'Variable': var,
                    'Mean_Treated': mean_treated,
                    'Mean_Control_Weighted': mean_control_weighted,
                    'N_Obs': len(temp_df)
                })

            if not desc_stats:
                continue

            desc_df = pd.DataFrame(desc_stats)

            # 4. 合并回归结果与描述性统计
            final_df = pd.merge(reg_df, desc_df, on='Variable', how='left')

            # 5. 格式化列与计算相对变化率
            final_df['Label'] = final_df['Variable'].map(LABELS).fillna(final_df['Variable'])

            # 计算相对变化率 (Coefficient / Control_Mean)
            final_df['Relative_Change_%'] = (final_df['Coefficient'] / final_df['Mean_Control_Weighted']) * 100

            # 6. 整理最终输出列
            output_columns = [
                'Label',
                'Coefficient', 'Std_Error', 'P_Value',
                'CI_Lower', 'CI_Upper',
                'Mean_Treated', 'Mean_Control_Weighted', 'Relative_Change_%',
                'N_Obs'
            ]

            final_table = final_df[output_columns].copy()

            numeric_cols = [
                'Coefficient', 'Std_Error', 'P_Value',
                'CI_Lower', 'CI_Upper',
                'Mean_Treated', 'Mean_Control_Weighted', 'Relative_Change_%'
            ]

            # 使用 apply 和 format 确保输出为 "0.00" 格式的字符串
            # 这样即使是 0.5 也会显示为 0.50
            for col in numeric_cols:
                if col in final_table.columns:
                    final_table[col] = final_table[col].apply(lambda x: f"{x:.2f}")

            # 添加星号标记 (注意：P_Value 现在是字符串，需要转回 float 进行比较，或者在转字符串之前计算 Sig)
            # 为了安全起见，我们重新从原始 final_df 获取 P值来计算星号
            final_table['Sig'] = final_df['P_Value'].apply(
                lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            )

            # 重新排序列，把 Sig 放在 P_Value 后面
            cols = list(final_table.columns)
            if 'P_Value' in cols and 'Sig' in cols:
                cols.insert(cols.index('P_Value') + 1, cols.pop(cols.index('Sig')))
            final_table = final_table[cols]

            # 7. 保存
            safe_country = country.replace(" ", "_")
            save_path = os.path.join(self.result_dir, f'Table_Detail_{safe_country}.csv')
            final_table.to_csv(save_path, index=False)

    def summarize_and_save(self):
        print("\n" + "=" * 60 + "\n7. Analysis and Summary\n" + "=" * 60)

        n_balanced = sum(self.balance_results['Is_Balanced']) if self.balance_results is not None else 0
        print("\n1. Matching Summary:")
        print(f"  Original samples: {len(self.df)}")
        print(f"  Matched samples: {len(self.matched_df)}")

        print(f"\n2. Balance Summary:")
        print(f"  Covariates: {len(self.covariates)}")
        print(f"  Balanced: {n_balanced} (SMD < 0.1)")

        print(f"\n3. Within-Country Regression Summary:")
        if hasattr(self, 'within_country_results') and not self.within_country_results.empty:
            total = len(self.within_country_results)
            sig05 = self.within_country_results['Significant_05'].sum()
            sig10 = self.within_country_results['Significant_10'].sum()
            print(f"  Total (country, variable) pairs: {total}")
            print(f"  Significant at p<0.05: {sig05}")
            print(f"  Significant at p<0.10: {sig10}")

            self.within_country_results.to_csv(
                os.path.join(self.result_dir, 'within_country_regression_results.csv'),
                index=False
            )
        else:
            print("  No within-country regression results.")

        self.matched_df.to_csv(os.path.join(self.result_dir, 'matched_data.csv'), index=False)
        if self.balance_results is not None:
            self.balance_results.to_csv(os.path.join(self.result_dir, 'balance_check.csv'), index=False)
        pd.DataFrame(self.balance_stats).to_csv(os.path.join(self.result_dir, 'country_stats.csv'), index=False)
        if self.country_smd_results:
            pd.DataFrame(self.country_smd_results).to_csv(
                os.path.join(self.result_dir, 'country_smd_results.csv'), index=False
            )


# ==========================================
# Main Entry Point
# ==========================================
def main():
    DATA_PATH = f'src/Results/Bias_Analysis/{THRESHOLD}/topsis/False/deployment_compatibility_topsis_entropy.csv'
    RESULT_DIR = f'src/Results/EB_Comp/{THRESHOLD}'

    COVARIATES = ['gdp', 'pop_density']
    TARGET_VARIABLES = ['annual_prep', 'snowfall', 'max_temp', 'min_temp', 'slope', 'road_density', 'sinuosity',
                        'ratio_complex', 'poi_richness']

    analysis = EntropyBalancingAnalysis(DATA_PATH, COVARIATES, TARGET_VARIABLES, LOG_TRANSFORM_VARIABLES,
                                        result_dir=RESULT_DIR)

    try:
        analysis.load_and_preprocess()
        analysis.run_entropy_balancing()
        analysis.check_balance()
        analysis.run_within_country_regressions()

        analysis.run_all_robustness_checks()

        print("\nGenerating visualizations...")
        analysis.plot_covariate_density()
        analysis.plot_love_plot()
        analysis.plot_faceted_forest_plot()
        analysis.plot_weight_diagnostics()
        analysis.plot_country_smd_heatmap()
        analysis.plot_overall_forest_plot()

        analysis.export_country_tables(target_countries=['China', 'United States'])
        analysis.summarize_and_save()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n❌ Analysis failed due to an error.")


if __name__ == "__main__":
    main()