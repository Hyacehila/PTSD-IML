# ============================================
# PTSD随机森林模型训练脚本
# 功能：构建随机森林模型，评估模型性能
# ============================================

# 加载必要的库
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(iml)
library(mlr3extralearners)

# 加载预处理后的数据
cat("加载预处理后的数据...\n")
load("data/PTSD_processed.RData")

# 检查数据是否正确加载
if (!exists("PTSD")) {
  stop("PTSD数据未找到，请先运行数据预处理脚本")
}

cat("数据维度:", dim(PTSD), "\n")
cat("目标变量PCLs的统计摘要:\n")
print(summary(PTSD$PCLs))

# 创建mlr3回归任务
# ============================================
cat("创建mlr3回归任务...\n")

task <- as_task_regr(PTSD, target = "PCLs", id = "PTSD_Prediction")

# 设置任务信息
task$set_row_roles(seq_len(nrow(PTSD)), role = "use")
cat("任务创建完成，特征数量:", task$nrow, "样本数:", task$ncol, "\n")

# 创建和配置随机森林学习器
# ============================================
cat("配置随机森林学习器...\n")

# 使用ranger随机森林学习器
learner <- lrn("regr.ranger")

# 设置超参数（可根据需要调整）
learner$param_set$values <- list(
  num.trees = 100,        # 树的数量
  mtry = NULL,            # 每次分裂考虑的特征数（NULL表示默认值）
  min.node.size = NULL,   # 最小节点大小
  importance = "permutation"  # 特征重要性计算方法
)

# 训练模型
# ============================================
cat("开始训练随机森林模型...\n")

# 训练模型
learner$train(task)

cat("模型训练完成！\n")

# 模型性能评估
# ============================================
cat("评估模型性能...\n")

# 在训练集上进行预测
prediction <- learner$predict(task)

# 计算性能指标
performance_metrics <- list(
  MSE = prediction$score(msr("regr.mse")),
  RMSE = prediction$score(msr("regr.rmse")),
  MAE = prediction$score(msr("regr.mae")),
  R2 = prediction$score(msr("regr.rsq"))
)

cat("模型性能指标:\n")
for (metric in names(performance_metrics)) {
  cat(sprintf("%s: %.4f\n", metric, performance_metrics[[metric]]))
}

# 保存模型
# ============================================
cat("保存训练好的模型...\n")

# 保存学习器对象
save(learner, file = "models/rf_model.RData")
cat("模型已保存到 models/rf_model.RData\n")

# 输出模型信息
cat("\n模型训练摘要:\n")
cat("- 算法: 随机森林 (ranger)\n")
cat("- 树的数量:", learner$param_set$values$num.trees, "\n")
cat("- 特征数量:", length(task$feature_names), "\n")
cat("- 训练样本数:", task$nrow, "\n")
cat("- 均方误差 (MSE):", round(performance_metrics$MSE, 4), "\n")
cat("- 决定系数 (R²):", round(performance_metrics$R2, 4), "\n")

# 返回训练好的学习器和性能指标
invisible(list(
  learner = learner,
  task = task,
  performance = performance_metrics
