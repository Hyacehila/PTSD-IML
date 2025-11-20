# ============================================
# PTSD IML可解释性分析脚本
# 功能：基于现有数据进行的完整IML分析
# ============================================

# 确保PTSD数据已加载（从预处理后的数据或原始数据）
if (!exists("PTSD")) {
  cat("加载预处理后的数据...\n")
  load("data/PTSD_processed.RData")
  if (!exists("PTSD")) {
    stop("PTSD数据未找到，请先运行数据预处理脚本")
  }
}

# 如果数据包含Age列，移除它（与IML正文部分保持一致）
if ("Age" %in% colnames(PTSD)) {
  PTSD$Age <- NULL
}

# 设置列名（与IML正文部分保持一致）
colnames(PTSD) <- c("Sex", "Place_of_Origin","Level_of_Education","Marital",
                  "Average_Income","Incidents","Incidents_Family",
                  "Witnessed_Injuries","body","Disaster_Scenes",
                  "Psychiatric_History","Use_of_Medications",
                  "Smoking_Status","Alcohol","ASDs","PCLs",
                  "Psychological_Resilience","Genetic_History")

# 移除ASD性质列（如果存在）
if ("ASD性质" %in% colnames(PTSD)) {
  PTSD$ASD性质 <- NULL
}

cat("数据准备完成，当前数据维度:", dim(PTSD), "\n")

# 加载必要的库
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(iml)
library(mlr3extralearners)

# 创建mlr3回归任务
# ============================================
cat("创建mlr3回归任务...\n")
task = as_task_regr(PTSD, target = "PCLs", id = "PTSD")

# 选择随机森林学习器
learner = lrn("regr.ranger")

# 训练模型
cat("开始训练随机森林模型...\n")
learner$train(task)
cat("模型训练完成！\n")

# 创建预测模型对象
predictor = Predictor$new(learner, data = PTSD, y = PTSD$PCLs)

# 计算置换特征重要性
# ============================================
cat("计算置换特征重要性...\n")
imp = FeatureImp$new(predictor, loss = "mse")

# 打印结果
cat("\n=== 置换特征重要性结果 ===\n")
print(imp)

# 绘制重要性图
if (!dir.exists("results")) {
  dir.create("results")
}
png("results/feature_importance_iml.png", width = 800, height = 600, res = 150)
plot(imp)
dev.off()
cat("特征重要性图已保存到 results/feature_importance_iml.png\n")

# 集群交互效应
# ============================================
cat("\n计算集群交互效应...\n")
interaction = Interaction$new(predictor)

# 打印交互效应
cat("\n=== 集群交互效应结果 ===\n")
print(interaction)

# 绘制交互效应图
png("results/interaction_effects_iml.png", width = 800, height = 600, res = 150)
plot(interaction)
dev.off()
cat("集群交互效应图已保存到 results/interaction_effects_iml.png\n")

# 个体特征交互效应分析
# ============================================
cat("\n分析个体特征的交互效应...\n")
cat("根据原文，ASDs、Psychological_Resilience、Disaster_Scenes 交互比较强，\n")
cat("超过了百分之10的方差解释度，考虑对他们分析个体交互\n")

# 考虑个体ASDs
cat("\n分析ASDs的交互效应...\n")
interaction_ASDs = Interaction$new(predictor, feature = "ASDs")

# 打印特定特征的交互效应
print(interaction_ASDs)

# 绘制特定特征的交互效应图
png("results/interaction_ASDs_iml.png", width = 800, height = 600, res = 150)
plot(interaction_ASDs)
dev.off()
cat("ASDs交互效应图已保存到 results/interaction_ASDs_iml.png\n")

# Psychological_Resilience
cat("\n分析Psychological_Resilience的交互效应...\n")
interaction_Psychological_Resilience = Interaction$new(predictor, feature = "Psychological_Resilience")

# 打印特定特征的交互效应
print(interaction_Psychological_Resilience)

# 绘制特定特征的交互效应图
png("results/interaction_Psychological_Resilience_iml.png", width = 800, height = 600, res = 150)
plot(interaction_Psychological_Resilience)
dev.off()
cat("Psychological_Resilience交互效应图已保存到 results/interaction_Psychological_Resilience_iml.png\n")

# Disaster_Scenes
cat("\n分析Disaster_Scenes的交互效应...\n")
interaction_Disaster_Scenes = Interaction$new(predictor, feature = "Disaster_Scenes")

# 打印特定特征的交互效应
print(interaction_Disaster_Scenes)

# 绘制特定特征的交互效应图
png("results/interaction_Disaster_Scenes_iml.png", width = 800, height = 600, res = 150)
plot(interaction_Disaster_Scenes)
dev.off()
cat("Disaster_Scenes交互效应图已保存到 results/interaction_Disaster_Scenes_iml.png\n")

# ALE分析
# ============================================
cat("\n进行ALE分析...\n")

# ASDs的ALE效应
cat("计算ASDs的ALE效应...\n")
ale = FeatureEffect$new(predictor, feature = "ASDs", method = "ale")

# 打印 ALE 结果
print(ale)

# 绘制 ALE 图
png("results/ale_ASDs_iml.png", width = 800, height = 600, res = 150)
plot(ale)
dev.off()
cat("ASDs的ALE图已保存到 results/ale_ASDs_iml.png\n")

# PDP分析
# ============================================
cat("\n进行PDP分析...\n")

# ASDs的PDP
cat("计算ASDs的PDP...\n")
pdp = FeatureEffect$new(predictor, feature = "ASDs", method = "pdp")

# 打印 PDP 结果
print(pdp)

# 绘制 PDP 图
png("results/pdp_ASDs_iml.png", width = 800, height = 600, res = 150)
plot(pdp)
dev.off()
cat("ASDs的PDP图已保存到 results/pdp_ASDs_iml.png\n")

# ASDs和Psychological_Resilience的交互PDP
cat("计算ASDs和Psychological_Resilience的交互PDP...\n")
pdp_interaction = FeatureEffect$new(predictor, feature = c("ASDs","Psychological_Resilience"), method = "pdp")

# 打印 PDP 结果
print(pdp_interaction)

# 绘制 PDP 图
png("results/pdp_ASDs_Psychological_Resilience_iml.png", width = 800, height = 600, res = 150)
plot(pdp_interaction)
dev.off()
cat("ASDs和Psychological_Resilience的PDP图已保存到 results/pdp_ASDs_Psychological_Resilience_iml.png\n")

# SHAP分析
# ============================================
cat("\n进行SHAP分析...\n")

# 创建SHAP解释器（使用第一个样本）
cat("创建SHAP解释器...\n")
shapley = Shapley$new(predictor, x.interest = PTSD[1,])

# 计算SHAP值
shapley$explain(x.interest = PTSD[1,])

# 打印结果
print(shapley)

# 绘制SHAP图
png("results/shap_sample1_iml.png", width = 800, height = 600, res = 150)
plot(shapley)
dev.off()
cat("第一个样本的SHAP图已保存到 results/shap_sample1_iml.png\n")

# 保存模型和结果
# ============================================
cat("\n保存模型和结果...\n")

# 保存模型
save(learner, file = "models/rf_model_iml.RData")
cat("模型已保存到 models/rf_model_iml.RData\n")

# 保存重要的分析结果
iml_results <- list(
  feature_importance = imp$results,
  interaction = interaction$results,
  interaction_ASDs = interaction_ASDs$results,
  interaction_Psychological_Resilience = interaction_Psychological_Resilience$results,
  interaction_Disaster_Scenes = interaction_Disaster_Scenes$results,
  ale_ASDs = ale$results,
  pdp_ASDs = pdp$results,
  pdp_interaction = pdp_interaction$results,
  shapley = shapley$results
)

save(iml_results, file = "results/iml_results.RData")
cat("IML分析结果已保存到 results/iml_results.RData\n")

# 生成分析摘要
# ============================================
cat("\n=== IML分析摘要 ===\n")

# 特征重要性摘要
cat("特征重要性前5名:\n")
top_importance <- head(imp$results[order(-imp$results$.imp), ], 5)
for (i in 1:nrow(top_importance)) {
  cat(sprintf("%d. %s: %.4f\n", i, top_importance$feature[i], top_importance$.imp[i]))
}

# 高交互效应特征
cat("\n高交互效应特征（>0.1）:\n")
high_interaction <- interaction$results[interaction$results$.imp > 0.1, ]
if (nrow(high_interaction) > 0) {
  for (i in 1:nrow(high_interaction)) {
    cat(sprintf("%d. %s: %.4f\n", i, high_interaction$feature[i], high_interaction$.imp[i]))
  }
} else {
  cat("未发现超过0.1的交互效应\n")
}

cat("\n所有IML分析完成！\n")
cat("生成的文件：\n")
cat("- models/rf_model_iml.RData: 训练好的模型\n")
cat("- results/iml_results.RData: 完整的分析结果\n")
cat("- results/*_iml.png: 各种分析图表\n")

# 返回结果对象
invisible(list(
  learner = learner,
  predictor = predictor,
  imp = imp,
  interaction = interaction,
  iml_results = iml_results
