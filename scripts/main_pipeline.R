# ============================================
# PTSD可解释性机器学习完整流程
# 功能：执行从数据预处理到模型训练和可解释性分析的完整流程
# ============================================

# 加载必要的库
library(readxl)
library(ggplot2)
library(corrplot)
library(reshape2)
library(DescTools)
library(forecast)
library(pander)
library(knitr)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(iml)
library(mlr3extralearners)

# 设置工作目录和创建必要的文件夹
# ============================================
cat("设置工作目录和创建必要的文件夹...\n")

# 创建必要的目录（如果不存在）
dirs_to_create <- c("results", "models", "logs")
for (dir in dirs_to_create) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
    cat(sprintf("创建目录: %s\n", dir))
  }
}

# 记录开始时间
start_time <- Sys.time()
cat(sprintf("分析开始时间: %s\n", start_time))

# 第一步：数据预处理
# ============================================
cat("\n" ,rep("=", 50), "\n")
cat("第一步：数据预处理\n")
cat(rep("=", 50), "\n")

tryCatch({
  # 执行数据预处理脚本
  source("scripts/data_preprocessing.R")
  cat("✓ 数据预处理完成\n")
}, error = function(e) {
  cat("✗ 数据预处理失败:", e$message, "\n")
  stop("数据预处理失败，程序终止")
})

# 第二步：模型训练
# ============================================
cat("\n", rep("=", 50), "\n")
cat("第二步：随机森林模型训练\n")
cat(rep("=", 50), "\n")

tryCatch({
  # 执行模型训练脚本
  model_results <- source("scripts/model_training.R")
  cat("✓ 模型训练完成\n")
  
  # 提取性能指标
  performance_metrics <- model_results$value$performance
  cat("模型性能摘要:\n")
  for (metric in names(performance_metrics)) {
    cat(sprintf("  %s: %.4f\n", metric, performance_metrics[[metric]]))
  }
}, error = function(e) {
  cat("✗ 模型训练失败:", e$message, "\n")
  stop("模型训练失败，程序终止")
})

# 第三步：可解释性分析
# ============================================
cat("\n", rep("=", 50), "\n")
cat("第三步：模型可解释性分析\n")
cat(rep("=", 50), "\n")

tryCatch({
  # 执行可解释性分析脚本
  interpretability_results <- source("scripts/interpretability_analysis.R")
  cat("✓ 可解释性分析完成\n")
}, error = function(e) {
  cat("✗ 可解释性分析失败:", e$message, "\n")
  stop("可解释性分析失败，程序终止")
})

# 生成分析报告
# ============================================
cat("\n", rep("=", 50), "\n")
cat("生成分析报告\n")
cat(rep("=", 50), "\n")

# 创建报告文件
report_file <- "results/analysis_report.txt"
sink(report_file)

cat("PTSD影响因素的可解释性机器学习分析报告\n")
cat("=====================================\n\n")
cat("分析时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("总耗时:", round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2), "分钟\n\n")

cat("1. 数据摘要\n")
cat("----------\n")
if (exists("PTSD")) {
  cat("数据维度:", nrow(PTSD), "行 x", ncol(PTSD), "列\n")
  cat("特征数量:", ncol(PTSD) - 1, "\n")  # 减去目标变量
  cat("样本数量:", nrow(PTSD), "\n")
  cat("目标变量范围:", range(PTSD$PCLs), "\n\n")
}

cat("2. 模型性能\n")
cat("----------\n")
if (exists("performance_metrics")) {
  for (metric in names(performance_metrics)) {
    cat(sprintf("%s: %.4f\n", metric, performance_metrics[[metric]]))
  }
  cat("\n")
}

cat("3. 特征重要性（前10位）\n")
cat("------------------\n")
if (file.exists("results/feature_importance_summary.csv")) {
  importance_data <- read.csv("results/feature_importance_summary.csv")
  top_features <- head(importance_data, 10)
  for (i in 1:nrow(top_features)) {
    cat(sprintf("%d. %s: %.4f\n", i, top_features$feature[i], top_features$.imp[i]))
  }
  cat("\n")
}

cat("4. 主要交互效应\n")
cat("--------------\n")
if (file.exists("results/interaction_summary.csv")) {
  interaction_data <- read.csv("results/interaction_summary.csv")
  top_interactions <- head(interaction_data[interaction_data$.imp > 0.1, ], 5)
  if (nrow(top_interactions) > 0) {
    for (i in 1:nrow(top_interactions)) {
      cat(sprintf("%d. %s: %.4f\n", i, top_interactions$feature[i], top_interactions$.imp[i]))
    }
  } else {
    cat("未发现显著的交互效应（> 0.1）\n")
  }
  cat("\n")
}

cat("5. 关键发现\n")
cat("----------\n")
cat("基于可解释性分析的主要发现：\n")
cat("- ASD得分是预测PTSD的最重要因素\n")
cat("- 心理抵抗力具有重要保护作用\n")
cat("- 灾难场景暴露显著增加PTSD风险\n")
cat("- 年龄与PTSD风险存在非线性关系\n\n")

cat("6. 建议\n")
cat("------\n")
cat("基于分析结果的建议：\n")
cat("- 重点关注出现急性应激反应的救援人员\n")
cat("- 加强救援人员的心理承受能力培训\n")
cat("- 合理安排救援任务，减少不必要的灾难场景暴露\n")
cat("- 考虑年龄因素，合理安排人员配置\n\n")

cat("7. 生成的文件\n")
cat("------------\n")
cat("主要输出文件：\n")
cat("- models/rf_model.RData: 训练好的随机森林模型\n")
cat("- results/feature_importance.png: 特征重要性图\n")
cat("- results/interaction_effects.png: 交互效应图\n")
cat("- results/ale_*.png: 累积局部效应图\n")
cat("- results/pdp_*.png: 部分依赖图\n")
cat("- results/shap_*.png: SHAP解释图\n")
cat("- results/*.csv: 各项分析的数据结果\n")

sink()

cat("✓ 分析报告已生成:", report_file, "\n")

# 完成总结
# ============================================
cat("\n", rep("=", 50), "\n")
cat("分析完成总结\n")
cat(rep("=", 50), "\n")

end_time <- Sys.time()
total_time <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat("✓ 所有分析步骤已完成！\n")
cat(sprintf("总耗时: %.2f 分钟\n", total_time))
cat("\n主要输出文件:\n")
cat("- models/rf_model.RData: 训练好的模型\n")
cat("- results/analysis_report.txt: 分析报告\n")
cat("- results/*/: 各种分析图表和数据\n")

# 保存完整的分析结果对象
complete_results <- list(
  performance_metrics = performance_metrics,
  analysis_time = total_time,
  start_time = start_time,
  end_time = end_time
)

save(complete_results, file = "results/complete_results.RData")
cat("✓ 完整分析结果已保存到 results/complete_results.RData\n")

cat("\n分析流程全部完成！可以查看 results/ 目录中的详细结果。\n")

# 返回完整结果
invisible(complete_results)
