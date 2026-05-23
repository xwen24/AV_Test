# ==============================================================================
# GLMM 分析管道 - 主模型 + 稳健性 + Moran's I
# ==============================================================================

library(tidyverse)
library(data.table)
library(lme4)       
library(car)        
library(broom.mixed)
library(sf)
library(spdep)

# ------------------------------------------------------------------------------
# 1. 配置路径和参数
# ------------------------------------------------------------------------------
current_wd <- getwd()
GRID_RESOLUTION_METERS <- 1000

RESULTS_DIR <- file.path(current_wd, paste0("results/Intra_City/Grid_", GRID_RESOLUTION_METERS))
GEOJSON_DIR <- file.path(RESULTS_DIR, "Grid")

# 变量标签映射
LABELS <- c(
  "gdp" = "GDP per capita",
  "gdp_log" = "log GDP per capita",
  "gdp_sum" = "Total GDP",
  "pop_density" = "Population density",
  "pop_density_log" = "log Population density",
  "pop_size" = "Population size",
  "ntl_mean" = "NTL intensity",
  "ntl_mean_log" = "log NTL intensity",
  "annual_prep" = "Annual rainfall",
  "annual_prep_log" = "log Annual rainfall",
  "extreme_prep" = "Annual days with rainfall>20mm/d",
  "snowfall" = "Annual snowfall",
  "snowfall_log" = "log Annual snowfall",
  "max_temp" = "Annual days above 40°C",
  "max_temp_log" = "log Annual days above 40°C",
  "min_temp" = "Annual days below -10°C",
  "min_temp_log" = "log Annual days below -10°C",
  "slope" = "Slope",
  "entropy" = "Road network orientation entropy",
  "road_density" = "Road density",
  "sinuosity" = "Road sinuosity",
  "ratio_complex" = "Complex intersection ratio",
  "dist_to_station" = "Distance to nearest station",
  "poi_richness" = "POI richness",
  "poi_richness_log" = "log POI richness",
  "poi_entropy" = "POI Shannon entropy"
)

# LABELS <- c(
#   "gdp" = "GDP per capita (US$)",
#   "gdp_log" = "log GDP per capita (US$)",
#   "gdp_sum" = "Total GDP (US$)",
#   "pop_density" = "Population density (/km$^2$)",
#   "pop_density_log" = "log Population density (/km$^2$)",
#   "pop_size" = "Population size",
#   "ntl_mean" = "NTL intensity (nW cm$^{-2}$ sr$^{-1}$)",
#   "ntl_mean_log" = "log NTL intensity (nW cm$^{-2}$ sr$^{-1}$)",
#   "annual_prep" = "Annual rainfall (mm)",
#   "annual_prep_log" = "log Annual rainfall (mm)",
#   "extreme_prep" = "Annual days with rainfall>20mm/d",
#   "snowfall" = "Annual snowfall (mm)",
#   "snowfall_log" = "log Annual snowfall (mm)",
#   "max_temp" = "Annual days above 40°C",
#   "max_temp_log" = "log Annual days above 40°C",
#   "min_temp" = "Annual days below -10°C",
#   "min_temp_log" = "log Annual days below -10°C",
#   "slope" = "Slope (degree)",
#   "entropy" = "Road network orientation entropy",
#   "road_density" = "Road density (km/km$^2$)",
#   "sinuosity" = "Road sinuosity",
#   "ratio_complex" = "Complex intersection ratio (%)",
#   "dist_to_station": "Distance to nearest station (km)",
#   "poi_richness" = "POI richness",
#   "poi_richness_log" = "log POI richness",
#   "poi_entropy" = "POI Shannon entropy"
# )

# ------------------------------------------------------------------------------
# 2. 数据加载与合并函数
# ------------------------------------------------------------------------------
load_and_merge_data <- function(x_vars, id_col = "grid_id", dep_var = "is_deployed") {
  if (length(x_vars) == 0) stop("自变量列表不能为空")
  if (!dir.exists(RESULTS_DIR)) stop(paste("结果目录不存在:", RESULTS_DIR))
  
  var_dirs <- list.dirs(RESULTS_DIR, full.names = FALSE, recursive = FALSE)
  var_dirs <- var_dirs[var_dirs != ""]
  
  missing_dirs <- setdiff(x_vars, var_dirs)
  if (length(missing_dirs) > 0) {
    warning(paste("以下变量目录不存在:", paste(missing_dirs, collapse = ", ")))
    x_vars <- intersect(x_vars, var_dirs)
  }
  if (length(x_vars) == 0) stop("没有找到有效的变量目录")
  
  first_var_path <- file.path(RESULTS_DIR, x_vars[1])
  city_files <- list.files(first_var_path, pattern = "^.*\\.csv$", full.names = FALSE)
  cities <- tools::file_path_sans_ext(city_files)
  
  all_data_list <- list()
  for (city in cities) {
    city_df <- NULL
    for (i in seq_along(x_vars)) {
      var_name <- x_vars[i]
      file_path <- file.path(RESULTS_DIR, var_name, paste0(city, ".csv"))
      if (!file.exists(file_path)) next 
      dt_temp <- tryCatch(fread(file_path), error = function(e) NULL)
      if (is.null(dt_temp)) next
      required_cols <- c(id_col, var_name)
      if (i == 1) required_cols <- c(required_cols, dep_var)
      dt_subset <- dt_temp[, ..required_cols]
      if (is.null(city_df)) {
        city_df <- dt_subset
        city_df[, city_name := city]
      } else {
        city_df <- merge(city_df, dt_subset, by = id_col, all = TRUE) 
      }
    }
    if (!is.null(city_df)) all_data_list[[city]] <- city_df
  }
  if (length(all_data_list) == 0) return(NULL)
  return(as.data.frame(rbindlist(all_data_list, fill = TRUE)))
}

# ------------------------------------------------------------------------------
# 3. 城市内标准化
# ------------------------------------------------------------------------------
standardize_within_city <- function(df, x_vars) {
  df %>%
    group_by(city_name) %>%
    mutate(across(all_of(x_vars), ~ {
      valid_values <- .x[!is.na(.x)]
      if (length(valid_values) < 10) return(rep(NA, length(.x)))
      m <- mean(valid_values, na.rm = TRUE)
      s <- sd(valid_values, na.rm = TRUE)
      if (is.na(s) || s == 0) s <- 1
      (.x - m) / s
    })) %>%
    ungroup()
}

# ------------------------------------------------------------------------------
# 4. VIF 检查
# ------------------------------------------------------------------------------
check_vif <- function(df, formula_str, output_path = NULL, threshold = 10) {
  message("------------------------------")
  message("正在进行多重共线性检查 (VIF)...")
  if (any(is.na(df))) warning("数据中存在缺失值，VIF计算可能不准确")
  
  tryCatch({
    dummy_model <- glm(as.formula(formula_str), data = df, family = binomial)
    if (!dummy_model$converged) warning("GLM模型未收敛，VIF结果可能不可靠")
    
    vif_vals <- car::vif(dummy_model)
    vif_df <- data.frame(
      Variable = names(vif_vals),
      VIF = as.numeric(vif_vals),
      Status = ifelse(vif_vals > threshold, "High", "OK")
    )
    
    # 打印到控制台
    print(vif_df)
    
    # 保存到 CSV
    if (!is.null(output_path)) {
      write.csv(vif_df, output_path, row.names = FALSE)
      message(sprintf("💾 VIF 结果已保存至: %s", output_path))
    }
    
    high_vif <- vif_vals[vif_vals > threshold]
    if (length(high_vif) > 0) {
      message(sprintf("\n⚠️ 警告: %d 个变量的VIF > %d", length(high_vif), threshold))
      message("高VIF变量: ", paste(names(high_vif), collapse = ", "))
    } else {
      message("\n✅ VIF检查通过 (所有变量VIF < ", threshold, ")")
    }
    
    return(vif_df)
    
  }, error = function(e) {
    warning("VIF计算失败: ", e$message)
    return(NULL)
  })
  message("------------------------------")
}

# ------------------------------------------------------------------------------
# 5. 计算主模型残差的 Moran's I
# ------------------------------------------------------------------------------
calculate_morans_i <- function(model, df_model, geojson_dir, id_col = "grid_id", model_label = "main") {
  message("Step 5.5: 计算 '", model_label, "' 模型残差的 Moran's I...")
  
  residuals_std <- residuals(model, type = "pearson")
  df_resid <- df_model %>% 
    select(city_name, all_of(id_col)) %>% 
    mutate(resid = residuals_std)
  
  moran_results <- tibble(city_name = character(), moran_i = numeric(), p_value = numeric())
  cities <- unique(df_resid$city_name)
  valid_count <- 0
  
  for (city in cities) {
    city_resid <- df_resid %>% filter(city_name == city)
    file_pattern <- paste0("^", city, "\\.geojson$")
    geo_path <- list.files(geojson_dir, pattern = file_pattern, full.names = TRUE, ignore.case = TRUE)
    
    if (length(geo_path) == 0) next
    
    tryCatch({
      city_sf <- st_read(geo_path, quiet = TRUE) %>%
        inner_join(city_resid, by = id_col)
      
      if (nrow(city_sf) <= 1) next
      
      nb <- poly2nb(city_sf, queen = TRUE, snap = 1e-4)
      if (all(sapply(nb, length) == 0)) next
      
      lw <- nb2listw(nb, style = "W", zero.policy = TRUE)
      moran_test <- moran.test(city_sf$resid, lw, zero.policy = TRUE)
      
      moran_results <- bind_rows(moran_results, tibble(
        city_name = city,
        moran_i = moran_test$estimate[1],
        p_value = moran_test$p.value
      ))
      valid_count <- valid_count + 1
      
    }, error = function(e) {
      next
    })
  }
  
  if (nrow(moran_results) > 0) {
    avg_moran <- mean(moran_results$moran_i, na.rm = TRUE)
    prop_sig <- mean(moran_results$p_value < 0.05, na.rm = TRUE)
    
    message(sprintf("✅ [%s] 平均 Moran's I = %.4f (n=%d cities)", model_label, avg_moran, valid_count))
    message(sprintf("✅ [%s] 显著比例 (p<0.05) = %.1f%%", model_label, 100 * prop_sig))
    
    # 用 model_label 区分输出文件
    output_file <- file.path(RESULTS_DIR, paste0("morans_i_results_", model_label, ".csv"))
    write.csv(moran_results, output_file, row.names = FALSE)
  } else {
    message("⚠️ [", model_label, "] 未计算任何 Moran's I（GeoJSON 缺失或拓扑错误）")
  }
}

# ------------------------------------------------------------------------------
# 6. 主执行函数
# ------------------------------------------------------------------------------
run_glmm_pipeline <- function(x_vars, dep_var = "is_deployed") {
  if (dir.exists(RESULTS_DIR)) {
    old_files <- list.files(RESULTS_DIR, pattern = "glmm_results_|morans_i|correlation_matrix", full.names = TRUE)
    if (length(old_files) > 0) {
      message("🧹 清理旧结果文件...")
      unlink(old_files)
    }
  }
  
  x_vars <- setdiff(x_vars, "spatial_lag")
  message("📌 主模型使用变量:", paste(x_vars, collapse = ", "))
  
  message("Step 1: Loading Data...")
  df <- load_and_merge_data(x_vars, dep_var = dep_var)
  
  message("Step 2: Cleaning Data...")
  df_clean <- df %>%
    drop_na(all_of(c(x_vars, dep_var, "city_name"))) %>%
    filter(!!sym(dep_var) %in% c(0, 1))
  
  valid_cities <- df_clean %>%
    group_by(city_name) %>%
    summarise(min_y = min(!!sym(dep_var)), max_y = max(!!sym(dep_var))) %>%
    filter(min_y == 0 & max_y == 1) %>%
    pull(city_name)
  
  df_clean <- df_clean %>% filter(city_name %in% valid_cities)
  
  # 部署比例
  deploy_count <- sum(df_clean[[dep_var]] == 1)
  total_grids <- nrow(df_clean)
  deploy_prop <- deploy_count / total_grids
  
  message(sprintf("📊 部署比例: %.2f%% (%d / %d)", 100 * deploy_prop, deploy_count, total_grids))
  message(sprintf("📈 有效城市数: %d", length(valid_cities)))
  
  # ------------------------------------------------------------------------------
  # 主模型：无 spatial_lag
  # ------------------------------------------------------------------------------
  message("Step 3: Fitting MAIN GLMM (no spatial_lag)...")
  df_model_main <- standardize_within_city(df_clean, x_vars)
  
  # 相关矩阵 & VIF
  cor_data <- df_model_main %>% select(all_of(x_vars))
  cor_matrix <- cor(cor_data, use = "complete.obs")
  cor_df <- as.data.frame(cor_matrix) %>% mutate(Variable = rownames(.)) %>% select(Variable, everything())
  write.csv(cor_df, file.path(RESULTS_DIR, "correlation_matrix_MAIN.csv"), row.names = FALSE)
  
  formula_str_main <- paste(dep_var, "~", paste(x_vars, collapse = " + "))
  check_vif(
    df_model_main, 
    formula_str_main, 
    output_path = file.path(RESULTS_DIR, "vif_results_MAIN.csv"), 
    threshold = 10
  )
  
  f_main <- as.formula(paste(dep_var, "~", paste(x_vars, collapse = " + "), "+ (1 | city_name)"))
  model_main <- glmer(f_main, data = df_model_main, family = binomial,
                      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)), nAGQ = 1)
  
  if (!is.null(model_main@optinfo$conv$opt) && model_main@optinfo$conv$opt != 0) {
    stop("主模型未收敛！")
  }
  message("✅ 主模型成功收敛")
  
  # 保存主模型结果
  res_main <- broom.mixed::tidy(model_main, effects = "fixed", conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      OR = exp(estimate),
      Lower_CI = exp(conf.low),
      Upper_CI = exp(conf.high),
      Label = LABELS[term]
    ) %>%
    select(term, Label, estimate, std.error, p.value, OR, Lower_CI, Upper_CI)
  write.csv(res_main, file.path(RESULTS_DIR, "glmm_results_MAIN.csv"), row.names = FALSE)
  
  # 调用诊断
  diagnose_glmm(
    model = model_main,
    data = df_model_main,
    dep_var = dep_var,
    model_label = "MAIN"
  )
  
  # 计算 Moran's I
  calculate_morans_i(model_main, df_model_main, GEOJSON_DIR, model_label = "MAIN")
  
  # ------------------------------------------------------------------------------
  # 稳健性检验：含 spatial_lag
  # ------------------------------------------------------------------------------
  message("Step 4: Fitting ROBUSTNESS GLMM (with spatial_lag)...")
  
  # 载入 spatial_lag
  # ------------------------------------------------------------------------------
  # 带缓存的空间滞后计算
  # ------------------------------------------------------------------------------
  source_spatial_lag <- function(df, geojson_dir, id_col = "grid_id", dep_var = "is_deployed") {
    message("🔄 计算空间滞后（带缓存）...")
    
    cities <- unique(df$city_name)
    df_list <- list()
    
    # 进度条
    if (requireNamespace("progress", quietly = TRUE)) {
      pb <- progress::progress_bar$new(
        total = length(cities),
        format = "计算空间滞后 [:bar] :percent (:current/:total) :eta"
      )
    }
    
    # 内存缓存：存储已读取的地理数据
    geo_cache <- new.env(parent = emptyenv())
    
    for (city in cities) {
      # 更新进度条
      if (exists("pb")) pb$tick()
      
      city_data <- df %>% filter(city_name == city)
      
      # 1. 检查缓存中是否有该城市的地理数据
      cache_key <- paste0("city_", city)
      
      if (!exists(cache_key, envir = geo_cache)) {
        # 2. 缓存未命中，从文件读取
        file_pattern <- paste0("^", city, "\\.geojson$")
        geo_path <- list.files(geojson_dir, pattern = file_pattern, 
                               full.names = TRUE, ignore.case = TRUE)
        
        if (length(geo_path) == 0) {
          message(sprintf("   ⚠️  %s: 未找到GeoJSON文件", city))
          city_data$spatial_lag <- NA
          df_list[[city]] <- city_data
          next
        }
        
        tryCatch({
          city_sf <- st_read(geo_path, quiet = TRUE) %>% 
            select(all_of(id_col))
          
          # 3. 存入缓存（只存储必要列）
          assign(cache_key, city_sf, envir = geo_cache)
          
        }, error = function(e) {
          message(sprintf("   ❌ %s: 读取GeoJSON失败 - %s", city, e$message))
          city_data$spatial_lag <- NA
          df_list[[city]] <- city_data
          return()
        })
      }
      
      # 4. 从缓存获取地理数据
      city_sf <- get(cache_key, envir = geo_cache)
      
      # 5. 计算空间滞后
      tryCatch({
        # 确保ID类型一致
        city_sf[[id_col]] <- as.character(city_sf[[id_col]])
        city_data[[id_col]] <- as.character(city_data[[id_col]])
        
        # 合并空间数据
        city_sf_joined <- city_sf %>% 
          inner_join(city_data, by = id_col)
        
        if (nrow(city_sf_joined) < 2) {
          message(sprintf("   ⚠️  %s: 有效网格数不足 (%d个)", city, nrow(city_sf_joined)))
          city_data$spatial_lag <- NA
          df_list[[city]] <- city_data
          return()
        }
        
        # 创建邻接矩阵
        nb <- poly2nb(city_sf_joined, queen = TRUE, snap = 1e-4)
        
        # 检查邻接关系
        if (all(sapply(nb, length) == 0)) {
          message(sprintf("   ⚠️  %s: 无邻接关系", city))
          city_data$spatial_lag <- NA
          df_list[[city]] <- city_data
          return()
        }
        
        # 计算空间滞后
        lw <- nb2listw(nb, style = "W", zero.policy = TRUE)
        y_vec <- as.numeric(city_sf_joined[[dep_var]])
        lag_vals <- lag.listw(lw, y_vec, zero.policy = TRUE)
        
        # 映射回原始数据
        lag_df <- data.frame(
          grid_id = city_sf_joined[[id_col]],
          spatial_lag = lag_vals
        )
        
        city_data <- city_data %>%
          left_join(lag_df, by = id_col)
        
        # 记录统计信息
        na_count <- sum(is.na(city_data$spatial_lag))
        message(sprintf("   ✅ %s: %d/%d 个网格有空间滞后值", 
                        city, nrow(city_data) - na_count, nrow(city_data)))
        
        df_list[[city]] <- city_data
        
      }, error = function(e) {
        message(sprintf("   ❌ %s: 空间滞后计算失败 - %s", city, e$message))
        city_data$spatial_lag <- NA
        df_list[[city]] <- city_data
      })
    }
    
    # 6. 清理缓存
    rm(geo_cache)
    
    # 7. 合并所有城市数据
    result <- bind_rows(df_list)
    
    # 输出汇总信息
    total_grids <- nrow(result)
    valid_lag_grids <- sum(!is.na(result$spatial_lag))
    message(sprintf("📊 空间滞后计算完成: %d/%d 个网格有有效值 (%.1f%%)",
                    valid_lag_grids, total_grids, 
                    100 * valid_lag_grids / total_grids))
    
    return(result)
  }
  
  df_robust <- source_spatial_lag(df_clean, GEOJSON_DIR, dep_var = dep_var) %>%
    drop_na(spatial_lag)
  df_robust$spatial_lag <- pmin(pmax(df_robust$spatial_lag, 0.05), 0.95)
  
  message(sprintf("🔄 稳健性模型：原始样本 %d → 有效 spatial_lag 样本 %d",
                  nrow(df_clean), nrow(df_robust)))
  
  x_vars_robust <- c(x_vars, "spatial_lag")
  df_model_robust <- standardize_within_city(df_robust, x_vars_robust)
  
  f_robust <- as.formula(paste(dep_var, "~", paste(x_vars_robust, collapse = " + "), "+ (1 | city_name)"))
  model_robust <- glmer(f_robust, data = df_model_robust, family = binomial,
                        control = glmerControl(
                          optimizer = "bobyqa",
                          optCtrl = list(maxfun = 200000),
                          check.conv.grad = .makeCC("warning", tol = 0.01),
                          nAGQ = 1
                        ), nAGQ = 1)
  
  if (!is.null(model_robust@optinfo$conv$opt) && model_robust@optinfo$conv$opt != 0) {
    message("⚠️ 未收敛")
  } else {
    message("✅ 稳健性模型成功收敛")
    
    res_robust <- broom.mixed::tidy(model_robust, effects = "fixed", conf.int = TRUE) %>%
      filter(term != "(Intercept)") %>%
      mutate(
        OR = exp(estimate),
        Lower_CI = exp(conf.low),
        Upper_CI = exp(conf.high),
        Label = c(LABELS, "spatial_lag" = "Spatial lag")[term]  # 扩展 LABELS
      ) %>%
      select(term, Label, estimate, std.error, p.value, OR, Lower_CI, Upper_CI)
    write.csv(res_robust, file.path(RESULTS_DIR, "glmm_results_ROBUST.csv"), row.names = FALSE)
    
    # 对稳健性模型计算 Moran's I
    message("🔍 计算稳健性模型残差的 Moran's I...")
    tryCatch({
      calculate_morans_i(model_robust, df_model_robust, GEOJSON_DIR, model_label = "ROBUST")
      # 将结果重命名或合并（见下方说明）
    }, error = function(e) {
      message("⚠️ 稳健性模型的 Moran's I 计算失败: ", e$message)
    })
    
    # 对稳健性模型进行诊断
    message("🔍 开始诊断 ROBUST 模型...")
    tryCatch({
      diagnose_glmm(
        model = model_robust,
        data = df_model_robust,
        dep_var = dep_var,
        model_label = "ROBUST"
      )
    }, error = function(e) {
      message("⚠️ ROBUST 模型诊断过程中出错: ", e$message)
    })
    
    # 保存预测概率用于 ROC 曲线绘图
    message("💾 Saving predicted probabilities for ROC visualization...")
    
    # 在 ROBUST 模型的样本上（交集）预测两个模型，均不包含随机效应
    pred_MAIN_on_ROBUST <- predict(model_main, newdata = df_model_robust, type = "response", re.form = NA)
    pred_ROBUST <- predict(model_robust, newdata = df_model_robust, type = "response", re.form = NA)
    
    predictions_df <- data.frame(
      city_name = df_model_robust$city_name,
      grid_id = df_model_robust$grid_id,
      y_true = df_model_robust[[dep_var]],
      pred_MAIN = pred_MAIN_on_ROBUST,
      pred_ROBUST = pred_ROBUST
    )
    
    write.csv(predictions_df, file.path(RESULTS_DIR, "predictions_for_ROC.csv"), row.names = FALSE)
    message("✅ Predictions saved to: predictions_for_ROC.csv")
  
  }
  
  message("✅ 分析完成！主结果见 glmm_results_MAIN.csv")
}

# ------------------------------------------------------------------------------
# 7. 模型诊断
# ------------------------------------------------------------------------------
diagnose_glmm <- function(model, data, dep_var, model_label) {
  message("🔍 Running enhanced GLMM diagnostics for: ", model_label, "...")
  
  diagnostics <- list()
  
  # 1. 收敛性与奇异拟合
  opt_conv <- FALSE
  if (!is.null(model@optinfo$conv$opt)) {
    opt_conv <- (model@optinfo$conv$opt == 0)
  } else if (!is.null(model@optinfo$conv$lme4)) {
    # For older lme4 versions
    opt_conv <- (model@optinfo$conv$lme4 == 0)
  }
  is_singular <- lme4::isSingular(model)
  diagnostics$converged <- as.numeric(opt_conv)
  diagnostics$singular_fit <- as.numeric(is_singular)
  
  message(sprintf("   -> Converged: %s | Singular fit: %s", 
                  ifelse(opt_conv, "Yes", "No"), 
                  ifelse(is_singular, "Yes", "No")))
  
  # 2. R-squared (Marginal & Conditional)
  diagnostics$Marginal_R2 <- NA
  diagnostics$Conditional_R2 <- NA
  
  if (requireNamespace("performance", quietly = TRUE)) {
    tryCatch({
      r2_out <- performance::r2(model, by_group = FALSE)
      diagnostics$Marginal_R2 <- r2_out$R2_marginal
      diagnostics$Conditional_R2 <- r2_out$R2_conditional
      message(sprintf("   -> R² (marginal/conditional): %.3f / %.3f", 
                      r2_out$R2_marginal, r2_out$R2_conditional))
    }, error = function(e) {
      message("⚠️ performance::r2() failed:", conditionMessage(e))
    })
  } else {
    message("ℹ️ 'performance' package not available → skipping R².")
  }
  
  # 3. AUC
  diagnostics$AUC <- NA
  if (requireNamespace("pROC", quietly = TRUE)) {
    tryCatch({
      pred_probs <- predict(model, type = "response", re.form = NA) # exclude RE for AUC
      auc_val <- pROC::auc(pROC::roc(data[[dep_var]], pred_probs))
      diagnostics$AUC <- as.numeric(auc_val)
      message(sprintf("   -> AUC: %.4f", auc_val))
    }, error = function(e) {
      message("⚠️ AUC calculation failed:", conditionMessage(e))
    })
  } else {
    message("ℹ️ 'pROC' not available → skipping AUC.")
  }
  
  # 4. DHARMa 残差诊断
  diagnostics$DHARMa_Dispersion_p <- NA
  diagnostics$DHARMa_Uniformity_p <- NA
  diagnostics$DHARMa_Outlier_p <- NA
  
  if (requireNamespace("DHARMa", quietly = TRUE)) {
    message("   -> Simulating DHARMa residuals (n=1000)...")
    tryCatch({
      sim_res <- DHARMa::simulateResiduals(
        fittedModel = model,
        n = 1000,
        plot = FALSE,
        integerResponse = TRUE
      )
      
      # Dispersion test
      disp_test <- tryCatch(
        DHARMa::testDispersion(sim_res, plot = FALSE),
        error = function(e) NULL
      )
      if (!is.null(disp_test)) diagnostics$DHARMa_Dispersion_p <- disp_test$p.value
      
      # Uniformity test (Kolmogorov-Smirnov)
      uni_test <- tryCatch(
        DHARMa::testUniformity(sim_res, plot = FALSE),
        error = function(e) NULL
      )
      if (!is.null(uni_test)) diagnostics$DHARMa_Uniformity_p <- uni_test$p.value
      
      # Outlier test (bootstrap)
      outlier_test <- tryCatch(
        DHARMa::testOutliers(sim_res, type = "bootstrap", nBoot = 100, plot = FALSE),
        error = function(e) NULL
      )
      if (!is.null(outlier_test) && !is.null(outlier_test$p.value)) {
        diagnostics$DHARMa_Outlier_p <- outlier_test$p.value
      }
      
      # Report key p-values
      message(sprintf("   -> DHARMa p-values: Dispersion=%.3f, Uniformity=%.3f, Outliers=%.3f",
                      diagnostics$DHARMa_Dispersion_p,
                      diagnostics$DHARMa_Uniformity_p,
                      diagnostics$DHARMa_Outlier_p))
      
    }, error = function(e) {
      message("⚠️ DHARMa simulation failed:", conditionMessage(e))
    })
  } else {
    message("ℹ️ 'DHARMa' not installed → skipping residual diagnostics.")
  }
  
  # 转换为 data.frame
  diag_df <- as.data.frame(t(unlist(diagnostics)), stringsAsFactors = FALSE)
  diag_df$model_name <- model_label
  
  # 保存
  output_path <- file.path(RESULTS_DIR, paste0("diagnostics_", model_label, ".csv"))
  write.csv(diag_df, output_path, row.names = FALSE)
  
  message("✅ Diagnostics saved to:", output_path)
  return(invisible(diag_df))
}

# ==============================================================================
# 运行
# ==============================================================================

my_vars <- c('ntl_mean_log', 'slope', 'road_density', 'sinuosity', 'ratio_complex', 'dist_to_station', 'poi_richness_log')
run_glmm_pipeline(my_vars)
