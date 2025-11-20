# ============================================
# PTSD数据预处理脚本
# 功能：数据导入、清洗、特征工程
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

# 数据导入和基础预处理
# ============================================
cat("开始数据导入和预处理...\n")

# 读取数据
PTSD <- read_excel("data/PTSD.xlsx", 
                   col_types = c("text", "numeric", "text", 
                                 "text", "text", "numeric", "numeric", 
                                 "text", "text", "text", "text", "text", 
                                 "text", "text", "text", "text", "text", 
                                 "text", "text", "text", "numeric", 
                                 "numeric", "numeric", "numeric", 
                                 "numeric", "text", "numeric", "numeric", 
                                 "numeric", "text"))

# 移除无用的ID列
PTSD <- PTSD[-1]

# 整理籍贯信息，只保留省级信息，避免特征数量过多
PTSD$籍贯 <- sub("-.*$", "", PTSD$籍贯)

# 检查缺失值
missing_count <- sum(is.na(PTSD))
cat("缺失值数量:", missing_count, "\n")

# 特征工程和变量筛选
# ============================================
cat("开始特征工程和变量筛选...\n")

# 1. 连续型数据分布分析
par(mfrow=c(2,3))
hist(PTSD$年龄, main = "Age Distribution")
hist(PTSD$身高, main = "Height Distribution") 
hist(PTSD$体重, main = "Weight Distribution")
hist(PTSD$BMI, main = "BMI Distribution")
hist(PTSD$ASD总分, main = "ASD Scores Distribution")
hist(PTSD$PCL总分, main = "PCL Scores Distribution")

# 2. BoxCox变换检查（虽然最后不采用）
as Lambda_ := BoxCox.lambda(PTSD$ASD总分, method = "guerrero")
pcl_lambda <- BoxCox.lambda(PTSD$PCL总分+1, method = "guerrero")
cat("ASD得分BoxCox lambda:", as_lambda, "\n")
cat("PCL得分BoxCox lambda:", pcl_lambda, "\n")

# 3. 相关性分析 - 移除不相关的连续变量
correlations <- data.frame(
  Variable = c("年龄", "身高", "体重", "BMI"),
  Correlation_with_PCL = c(
    cor.test(PTSD$年龄, PTSD$PCL总分)$estimate,
    cor.test(PTSD$身高, PTSD$PCL总分)$estimate,
    cor.test(PTSD$体重, PTSD$PCL总分)$estimate,
    cor.test(PTSD$BMI, PTSD$PCL总分)$estimate
  ),
  P_value = c(
    cor.test(PTSD$年龄, PTSD$PCL总分)$p.value,
    cor.test(PTSD$身高, PTSD$PCL总分)$p.value,
    cor.test(PTSD$体重, PTSD$PCL总分)$p.value,
    cor.test(PTSD$BMI, PTSD$PCL总分)$p.value
  )
)

print(correlations)

# 移除相关性很低的身高、体重、BMI指标
PTSD$身高 <- NULL
PTSD$体重 <- NULL
PTSD$BMI <- NULL

# 4. ASD子项得分分析
# 构建ASD解释模型
model.ASD.lm <- lm(PTSD$ASD总分 ~ 1 + PTSD$ASD分离 + PTSD$ASD再体验 +
                     PTSD$ASD回避 + PTSD$ASD警觉, data = PTSD)

cat("ASD总分模型R-squared:", summary(model.ASD.lm)$r.squared, "\n")

# 移除ASD子项得分（因为总分就是子项的线性组合）
PTSD$ASD分离 <- NULL
PTSD$ASD再体验 <- NULL
PTSD$ASD回避 <- NULL
PTSD$ASD警觉 <- NULL

# 5. ASD总分与ASD性质相关性
kruskal_result <- kruskal.test(PTSD$ASD总分 ~ PTSD$ASD性质, data = PTSD)
cat("ASD总分与性质Kruskal检验p值:", kruskal_result$p.value, "\n")

# 移除ASD性质（保留更精确的总分）
PTSD$ASD性质 <- NULL

# 6. 数据类型转换和因子化
# 有序因子
PTSD$文化程度 <- factor(PTSD$文化程度, ordered = TRUE, levels = c("3", "2", "1"))
PTSD$家庭人均月收入 <- factor(PTSD$家庭人均月收入, ordered = TRUE, levels = c("1", "2", "3", "4"))

# 将字符变量转换为因子
PTSD <- data.frame(lapply(PTSD, function(x) {
  if (is.character(x)) {
    as.factor(x)
  } else {
    x
  }
}))

# 7. 列名标准化（英文命名）
colnames(PTSD) <- c("Age", "Sex", "Place_of_Origin", "nation",
                    "Level_of_Education", "Marital",
                    "Average_Income", "Incidents", "Incidents_Family",
                    "Witnessed_Injuries", "body", "Disaster_Scenes",
                    "Psychiatric_History", "Use_of_Medications", "Smoking",
                    "Smoking_Status", "Alcohol", "ASDs", "PCLs",
                    "Psychological_Resilience", "Genetic_History")

# 8. 自相关性分析（使用Cramer's V系数）
cat("开始自相关性分析...\n")

factor_cols <- PTSD[, sapply(PTSD, is.factor)]
cramer_v_matrix <- matrix(0, ncol = ncol(factor_cols), nrow = ncol(factor_cols))
colnames(cramer_v_matrix) <- colnames(factor_cols)
rownames(cramer_v_matrix) <- colnames(factor_cols)

for (i in 1:ncol(factor_cols)) {
  for (j in 1:ncol(factor_cols)) {
    cramer_v_matrix[i, j] <- CramerV(table(factor_cols[, i], factor_cols[, j]))
  }
}

# 绘制相关性热力图
library(ggcorrplot)
ggcorrplot(cramer_v_matrix, lab = TRUE) +
  ggtitle("Cramer's V Correlation Matrix")

# 识别强相关变量（相关系数 > 0.7）
strong_corr <- which(cramer_v_matrix > 0.7 & cramer_v_matrix < 1, arr.ind = TRUE)
if (nrow(strong_corr) > 0) {
  cat("发现强相关变量对：\n")
  print(strong_corr)
}

# 移除颗粒度更低的吸烟指标（被Smoking_Status代替）
PTSD$Smoking <- NULL

# 保存预处理后的数据
save(PTSD, file = "data/PTSD_processed.RData")
cat("数据预处理完成，已保存到 data/PTSD_processed.RData\n")

# 输出数据摘要
cat("最终数据维度:", dim(PTSD), "\n")
