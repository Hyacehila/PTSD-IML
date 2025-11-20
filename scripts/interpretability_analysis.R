# ============================================
# PTSD模型可解释性分析脚本
# 功能：特征重要性、交互效应、ALE和PDP分析
# ============================================

# 加载必要的库
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(iml)
library(mlr3extralearners)
library(ggplot2)
library(pander)

# 加载训练好的模型和数据
# ============================================
cat("加载训练好的模型和数据...\n")

# 加载预处理后的数据
load("data/PTSD_processed.RData")
if (!exists("PTSD")) {
  stop("PTSD数据未找到，请先运行数据预处理脚本")
}

# 加载训练好的模型
load("models/rf_model.RData")
if (!exists("learner")) {
  stop("模型未找到，请先运行模型训练脚本")
}

cat("模型和数据加载完成！\n")

# 创建IML预测器对象
# ============================================
cat("创建IML预测器对象...\n")

predictor <- Predictor$new(
  learner, 
  data = PTSD[ , !(colnames(PTSD) %in% "PCLs")],  # 排除目标变量
  y = PTSD$PCLs
)

cat("预测器对象创建完成！\n")

# 1. 置换特征重要性分析
# ============================================
cat("计算置换特征重要性...\n")

# 计算置换特征重要性
feature_importance <- FeatureImp$new(predictor, loss = "mse")

# 打印结果
cat("\n=== 置换特征重要性结果 ===\n")
print(feature_importance$results)

# 绘制重要性图
png("results/feature_importance.png", width = 800, height = 600, res = 150)
plot(feature_importance)
dev.off()
cat("特征重要性图已保存到 results/feature_importance.png\n")

# 2. 弗里德曼H统计量 - 交互效应分析
# ============================================
cat("\n计算交互效应分析...\n")

# 总体交互效应
interaction <- Interaction$new(predictor)

cat("\n=== 集群交互效应结果 ===\n")
print(interaction$results)

# 绘制集群交互效应图
png("results/interaction_effects.png", width = 800, height = 600, res = 150)
plot(interaction)
dev.off()
cat("集群交互效应图已保存到 results/interaction_effects.png\n")

# 识别高交互效应的特征（方差解释度 > 20%）
high_interaction_features <- interaction$results[interaction$results$.imp > 0.2, ]
if (nrow(high_interaction_features) > 0) {
  cat("\n高交互效应特征（方差解释度 > 20%）:\n")
  print(high_interaction_features)
} else {
  cat("\n未发现方差解释度超过20%的交互效应\n")
}

# 3. 个体特征的交互效应分析
# ============================================
cat("\n分析个体特征的交互效应...\n")

# 选择重要特征进行详细分析
important_features <- c("ASDs", "Psychological_Resilience", "Disaster_Scenes", "Age")

for (feature in important_features) {
  if (feature %in% colnames(PTSD)) {
    cat(sprintf("\n分析特征 %s 的交互效应...\n", feature))
    
    # 计算个体交互效应
    feature_interaction <- Interaction$new(predictor, feature = feature)
    
    # 保存结果
    interaction_results <- feature_interaction$results
    write.csv(interaction_results, 
              file = sprintf("results/interaction_%s.csv", feature), 
              row.names = FALSE)
    
    # 绘图
    png(sprintf("results/interaction_%s.png", feature), 
        width = 800, height = 600, res = 150)
    plot(feature_interaction)
    dev.off()
    
    cat(sprintf("特征 %s 的交互效应分析完成\n", feature))
  }
}

# 4. 累积局部效应图 (ALE) 分析
# ============================================
cat("\n进行累积局部效应图分析...\n")

# 对重要特征进行ALE分析
ale_features <- c("ASDs", "Psychological_Resilience", "Disaster_Scenes", "Age")

for (feature in ale_features) {
  if (feature %in% colnames(PTSD)) {
    cat(sprintf("\n计算特征 %s 的ALE效应...\n", feature))
    
    # 计算ALE效应
    ale_effect <- FeatureEffect$new(predictor, feature = feature, method = "ale")
    
    # 保存ALE结果
    ale_results <- ale_effect$results
    write.csv(ale_results, 
              file = sprintf("results/ale_%s.csv", feature), 
              row.names = FALSE)
    
    # 绘制ALE图
    png(sprintf("results/ale_%s.png", feature), 
        width = 800, height = 600, res = 150)
    plot(ale_effect)
    dev.off()
    
    cat(sprintf("特征 %s 的ALE分析完成\n", feature))
  }
}

# 5. 二阶部分依赖图 (PDP) - 交互效应可视化
# ============================================
cat("\n进行二阶部分依赖图分析...\n")

# 定义重要的特征对进行交互分析
important_pairs <- list(
  c("ASDs", "Age"),
  c("ASDs", "Psychological_Resilience"),
  c("Disaster_Scenes", "Psychological_Resilience"),
  c("Age", "Smoking_Status")
)

for (pair in important_pairs) {
  # 检查特征是否存在
  if (all(pair %in% colnames(PTSD))) {
    cat(sprintf("\n分析特征对 %s 和 %s 的交互效应...\n", pair[1], pair[2]))
    
    # 计算二阶PDP
    pdp_effect <- FeatureEffect$new(predictor, feature = pair, method = "pdp")
    
    # 保存PDP结果
    pdp_results <- pdp_effect$results
    write.csv(pdp_results, 
              file = sprintf("results/pdp_%s_%s.csv", pair[1], pair[2]), 
              row.names = FALSE)
    
    # 绘制PDP图
    png(sprintf("results/pdp_%s_%s.png", pair[1], pair[2]), 
        width = 800, height = 600, res = 150)
    plot(pdp_effect)
    dev.off()
    
    cat(sprintf("特征对 %s 和 %s 的PDP分析完成\n", pair[1], pair[2]))
  }
}

# 6. SHAP分析 (可选)
# ============================================
cat("\n进行SHAP分析...\n")

# 选择几个样本进行SHAP分析
sample_indices <- c(1, min(10, nrow(PTSD)), min(20, nrow(PTSD)))

for (i in seq_along(sample_indices)) {
  idx <- sample_indices[i]
  cat(sprintf("分析第 %d 个样本的SHAP值...\n", idx))
  
  # 创建SHAP解释器
  shapley <- Shapley$new(predictor, x.interest = PTSD[idx, ])
  
  # 计算SHAP值
  shapley$explain(x.interest = PTSD[idx, ])
  
  # 保存SHAP结果
  shap_results <- shapley$results
  write.csv(shap_results, 
            file = sprintf("results/shap_sample_%d.csv", idx), 
            row.names = FALSE)
  
  # 绘制SHAP图
  png(sprintf("results/shap_sample_%d.png", idx), 
      width = 800, height = 600, res = 150)
  plot(shapley)
  dev.off()
  
  cat(sprintf("第 %d 个样本的SHAP分析完成\n", idx))
}

# 7. 结果总结
# ============================================
cat("\n=== 可解释性分析总结 ===\n")

# 保存特征重要性排序
importance_summary <- feature_importance$results[order(-feature_importance$results$.imp), ]
write.csv(importance_summary, "results/feature_importance_summary.csv", row.names = FALSE)

# 保存交互效应摘要
interaction_summary <- interaction$results[order(-interaction$results$.imp), ]
write.csv(interaction_summary, "results/interaction_summary.csv", row.names = FALSE)

cat("所有分析结果已保存到 results/ 目录\n")
cat("主要文件包括:\n")
cat("- feature_importance.png: 特征重要性图\n")
cat("- interaction_effects.png: 集群交互效应图\n")
cat("- ale_*.png: 各特征的累积局部效应图\n")
cat("- pdp_*.png: 特征对的二阶部分依赖图\n")
cat("- shap_sample_*.png: 样本的SHAP解释图\n")
cat("- feature_importance_summary.csv: 特征重要性数据\n")
cat("- interaction_summary.csv: 交互效应数据\n")

cat("\n可解释性分析完成！\n")

# 返回主要结果对象
invisible(list(
  predictor = predictor,
  feature_importance = feature_importance,
  interaction = interaction,
  importance_summary = importance_summary,
  interaction_summary = interaction_summary
))
