# ============================================================
# Synthetic clinical survival data generation script (Synthetic Clinical Survival Data)
# Purpose: demo data for the AI-Powered Clinical Data Explorer
# Uses base R only; the final validation section needs the survival package (optional)
# ============================================================

set.seed(2026)

# ---------- 1. Parameters ----------
n <- 1248                      # number of patients
follow_up_max <- 36            # maximum follow-up (months), administrative censoring
dropout_rate  <- 0.008         # random dropout rate (per month)
weib_shape    <- 0.8           # <1: hazard decreases over time, curve flattens late
baseline_lam  <- 0.037         # baseline scale (larger => worse overall survival)
add_missing   <- TRUE          # whether to add a small amount of missing pdl1, to test data validation

# ---------- 2. Baseline characteristics ----------
patient_id <- sprintf("P%04d", seq_len(n))

# Age: mostly over 65, truncated to 40-90
age <- round(pmin(pmax(rnorm(n, mean = 70, sd = 8), 40), 90))

sex <- sample(c("Female", "Male"), n, replace = TRUE, prob = c(0.52, 0.48))

# Stage I-IV
stage <- sample(c("I", "II", "III", "IV"), n, replace = TRUE,
                prob = c(0.10, 0.30, 0.38, 0.22))

# ECOG 0-2; older age makes higher ECOG more likely (mild correlation)
p_ecog2 <- plogis(-2.2 + 0.04 * (age - 70))
p_ecog1 <- plogis(-0.2 + 0.03 * (age - 70)) * (1 - p_ecog2)
p_ecog0 <- 1 - p_ecog1 - p_ecog2
ecog <- vapply(seq_len(n), function(i)
  sample(0:2, 1, prob = c(p_ecog0[i], p_ecog1[i], p_ecog2[i])), integer(1))

# Treatment groups (proportions match the screenshot: 36.2% / 32.7% / 31.1%)
treatment <- sample(c("A", "B", "C"), n, replace = TRUE,
                    prob = c(0.362, 0.327, 0.311))
treatment <- paste("Treatment", treatment)

# PD-L1 biomarker
pdl1 <- sample(c("Positive", "Negative"), n, replace = TRUE, prob = c(0.45, 0.55))

# Enrollment date (for the Trend Over Time chart)
enroll_date <- as.Date("2018-01-01") + sample(0:(365 * 5 + 180), n, replace = TRUE)

# ---------- 3. Survival time (Weibull proportional hazards model) ----------
# Linear predictor = log HR; treatment effect A < B < C (A is best)
lp <- 0 +
  c("Treatment A" = 0, "Treatment B" = 0.40, "Treatment C" = 0.85)[treatment] +
  0.15 * (age - 70) / 10 +
  c("I" = -0.60, "II" = 0, "III" = 0.45, "IV" = 0.90)[stage] +
  0.30 * ecog +
  ifelse(pdl1 == "Positive", -0.30, 0) +
  ifelse(sex == "Female", -0.10, 0)
lp <- as.numeric(lp)

# Inverse transform sampling: T = ( -log(U) / (lambda * exp(lp)) )^(1/shape)
u <- runif(n)
event_time <- (-log(u) / (baseline_lam * exp(lp)))^(1 / weib_shape)

# Censoring: random dropout + administrative censoring (36 months)
dropout_time  <- rexp(n, rate = dropout_rate)
censor_time   <- pmin(dropout_time, follow_up_max)

os_time  <- pmin(event_time, censor_time)
os_event <- as.integer(event_time <= censor_time)
os_time  <- round(os_time, 2)

# ---------- 3b. Progression-free survival (PFS) ----------
# PFS event = progression OR death, so pfs_time <= os_time always holds.
#   - dead patients: 85% had documented progression before death
#   - censored patients: 60% progressed before their last follow-up
# Progression happens at 30-95% of the OS time.
frac       <- runif(n, 0.3, 0.95)
progressed <- ifelse(os_event == 1, runif(n) < 0.85, runif(n) < 0.60)
pfs_time   <- round(ifelse(progressed, os_time * frac, os_time), 2)
pfs_event  <- as.integer(progressed | os_event == 1)

# ---------- 4. Assemble the data frame ----------
df <- data.frame(
  patient_id  = patient_id,
  age         = age,
  sex         = sex,
  stage       = stage,
  ecog        = ecog,
  treatment   = treatment,
  pdl1        = pdl1,
  enroll_date = enroll_date,
  os_time     = os_time,
  os_event    = os_event,
  pfs_time    = pfs_time,
  pfs_event   = pfs_event,
  stringsAsFactors = FALSE
)

# A few missing values (about 3% of pdl1)
if (add_missing) {
  miss_idx <- sample(n, round(0.03 * n))
  df$pdl1[miss_idx] <- NA
}

# ---------- 5. Save ----------
# Writes to the current working directory; run from the project root.
out_path <- "synthetic_clinical.csv"

write.csv(
  df,
  out_path,
  row.names = FALSE,
  na = ""
)

cat("Saved:", out_path, "(", nrow(df), "rows )\n")

# ---------- 6. Quick validation ----------
cat("\n--- Patients per treatment group ---\n")
print(table(df$treatment))
print(round(prop.table(table(df$treatment)) * 100, 1))

cat("\n--- Cohort with age > 65 ---\n")
sub <- subset(df, age > 65)
cat("N:", nrow(sub), " Median age:", median(sub$age),
    " Female proportion:", round(mean(sub$sex == "Female") * 100, 1), "%\n")

cat("\n--- Event rate ---\n")
print(round(tapply(df$os_event, df$treatment, mean), 3))
cat("\n--- PFS event rate ---\n")
print(round(tapply(df$pfs_event, df$treatment, mean), 3))

if (requireNamespace("survival", quietly = TRUE)) {
  library(survival)
  fit <- survfit(Surv(os_time, os_event) ~ treatment, data = sub)
  cat("\n--- 36-month survival probability (age > 65) ---\n")
  print(summary(fit, times = 36, extend = TRUE)$surv)
  lr <- survdiff(Surv(os_time, os_event) ~ treatment, data = sub)
  p  <- 1 - pchisq(lr$chisq, df = length(lr$n) - 1)
  cat("\nLog-rank p =", format.pval(p, digits = 3), "\n")
  
  # Plot a KM curve to check the shape
  plot(fit, col = c("blue", "purple", "forestgreen"), lwd = 2,
       xlab = "Time (months)", ylab = "Survival Probability",
       main = "Synthetic data check: OS by treatment (Age > 65)")
  legend("topright", legend = levels(factor(sub$treatment)),
         col = c("blue", "purple", "forestgreen"), lwd = 2, bty = "n")
} else {
  cat("\n(Install the survival package to see the KM check: install.packages('survival'))\n")
}
