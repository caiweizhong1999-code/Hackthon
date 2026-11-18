# ============================================================
# 合成临床数据生成脚本 (Synthetic Clinical Survival Data)
# 用途: AI-Powered Clinical Data Explorer 演示数据
# 仅使用 base R;最后的验证部分需要 survival 包(可选)
# ============================================================

set.seed(2026)

# ---------- 1. 参数 ----------
n <- 1248                      # 患者数
follow_up_max <- 36            # 最长随访(月),行政删失
dropout_rate  <- 0.008         # 随机失访速率(每月)
weib_shape    <- 0.8           # <1: 风险随时间下降,曲线后期趋平
baseline_lam  <- 0.037         # 基线尺度(调大 => 整体生存更差)
add_missing   <- TRUE          # 是否给 pdl1 加少量缺失,用来测试数据校验

# ---------- 2. 基线特征 ----------
patient_id <- sprintf("P%04d", seq_len(n))

# 年龄: 大部分在 65 岁以上,截断在 40-90
age <- round(pmin(pmax(rnorm(n, mean = 70, sd = 8), 40), 90))

sex <- sample(c("Female", "Male"), n, replace = TRUE, prob = c(0.52, 0.48))

# 分期 I-IV
stage <- sample(c("I", "II", "III", "IV"), n, replace = TRUE,
                prob = c(0.10, 0.30, 0.38, 0.22))

# ECOG 0-2,年龄越大越容易 ECOG 高(轻度相关)
p_ecog2 <- plogis(-2.2 + 0.04 * (age - 70))
p_ecog1 <- plogis(-0.2 + 0.03 * (age - 70)) * (1 - p_ecog2)
p_ecog0 <- 1 - p_ecog1 - p_ecog2
ecog <- vapply(seq_len(n), function(i)
  sample(0:2, 1, prob = c(p_ecog0[i], p_ecog1[i], p_ecog2[i])), integer(1))

# 治疗分组(比例对应截图: 36.2% / 32.7% / 31.1%)
treatment <- sample(c("A", "B", "C"), n, replace = TRUE,
                    prob = c(0.362, 0.327, 0.311))
treatment <- paste("Treatment", treatment)

# PD-L1 生物标志物
pdl1 <- sample(c("Positive", "Negative"), n, replace = TRUE, prob = c(0.45, 0.55))

# 入组日期(用于 Trend Over Time 图)
enroll_date <- as.Date("2018-01-01") + sample(0:(365 * 5 + 180), n, replace = TRUE)

# ---------- 3. 生存时间(Weibull 比例风险模型) ----------
# 线性预测子 = log HR,治疗效应为 A < B < C(A 最好)
lp <- 0 +
  c("Treatment A" = 0, "Treatment B" = 0.40, "Treatment C" = 0.85)[treatment] +
  0.15 * (age - 70) / 10 +
  c("I" = -0.60, "II" = 0, "III" = 0.45, "IV" = 0.90)[stage] +
  0.30 * ecog +
  ifelse(pdl1 == "Positive", -0.30, 0) +
  ifelse(sex == "Female", -0.10, 0)
lp <- as.numeric(lp)

# 逆变换抽样: T = ( -log(U) / (lambda * exp(lp)) )^(1/shape)
u <- runif(n)
event_time <- (-log(u) / (baseline_lam * exp(lp)))^(1 / weib_shape)

# 删失: 随机失访 + 行政删失(36 个月)
dropout_time  <- rexp(n, rate = dropout_rate)
censor_time   <- pmin(dropout_time, follow_up_max)

time_months <- pmin(event_time, censor_time)
event       <- as.integer(event_time <= censor_time)
time_months <- round(time_months, 2)

# ---------- 4. 组装数据框 ----------
df <- data.frame(
  patient_id  = patient_id,
  age         = age,
  sex         = sex,
  stage       = stage,
  ecog        = ecog,
  treatment   = treatment,
  pdl1        = pdl1,
  enroll_date = enroll_date,
  time_months = time_months,
  event       = event,
  stringsAsFactors = FALSE
)

# 少量缺失(约 3% 的 pdl1)
if (add_missing) {
  miss_idx <- sample(n, round(0.03 * n))
  df$pdl1[miss_idx] <- NA
}

# ---------- 5. 保存 ----------
dir.create("C:/Users/caiwei/Desktop/hackthon",
           showWarnings = FALSE,
           recursive = TRUE)

write.csv(
  df,
  "C:/Users/caiwei/Desktop/hackthon/synthetic_clinical.csv",
  row.names = FALSE,
  na = ""
)

cat("已保存: C:/Users/caiwei/Desktop/hackthon/synthetic_clinical.csv (",
    nrow(df), "行 )\n")

# ---------- 6. 快速验证 ----------
cat("\n--- 治疗组人数 ---\n")
print(table(df$treatment))
print(round(prop.table(table(df$treatment)) * 100, 1))

cat("\n--- 年龄 > 65 的队列 ---\n")
sub <- subset(df, age > 65)
cat("人数:", nrow(sub), " 年龄中位数:", median(sub$age),
    " 女性占比:", round(mean(sub$sex == "Female") * 100, 1), "%\n")

cat("\n--- 事件率 ---\n")
print(round(tapply(df$event, df$treatment, mean), 3))

if (requireNamespace("survival", quietly = TRUE)) {
  library(survival)
  fit <- survfit(Surv(time_months, event) ~ treatment, data = sub)
  cat("\n--- 36 个月生存概率(age > 65)---\n")
  print(summary(fit, times = 36, extend = TRUE)$surv)
  lr <- survdiff(Surv(time_months, event) ~ treatment, data = sub)
  p  <- 1 - pchisq(lr$chisq, df = length(lr$n) - 1)
  cat("\nLog-rank p =", format.pval(p, digits = 3), "\n")
  
  # 画一张 KM 图确认形状
  plot(fit, col = c("blue", "purple", "forestgreen"), lwd = 2,
       xlab = "Time (months)", ylab = "Survival Probability",
       main = "Synthetic data check: OS by treatment (Age > 65)")
  legend("topright", legend = levels(factor(sub$treatment)),
         col = c("blue", "purple", "forestgreen"), lwd = 2, bty = "n")
} else {
  cat("\n(安装 survival 包可查看 KM 验证: install.packages('survival'))\n")
}
