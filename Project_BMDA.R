# Load required packages
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(epiR)

# Load the dataset
data <- read.csv("C:/Users/fladn/Desktop/Rdataset/diabetes.csv", stringsAsFactors = FALSE)

# Replace 0s with NA in relevant columns
cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
data[cols] <- lapply(data[cols], function(x) ifelse(x == 0, NA, x))

# Fill NAs with median values
data[cols] <- lapply(data[cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Show summary stats
summary_stats <- data.frame(
  Variable = names(data),
  Mean = sapply(data, mean),
  SD = sapply(data, sd)
)
print("=== Summary Statistics ===")
print(summary_stats)

# Plot Distributions
setEPS()
postscript("bmi_distribution.eps", height = 6, width = 6, paper = "special", horizontal = FALSE)
ggplot(data, aes(x = BMI, fill = factor(Outcome))) +
  geom_histogram(position = "dodge", binwidth = 2) +
  theme_minimal() +
  labs(title = "BMI Distribution by Outcome", x = "BMI", fill = "Diabetes")
dev.off()

postscript("glucose_distribution.eps", height = 6, width = 6, paper = "special", horizontal = FALSE)
ggplot(data, aes(x = Glucose, fill = factor(Outcome))) +
  geom_histogram(position = "dodge", binwidth = 5) +
  theme_minimal() +
  labs(title = "Glucose Distribution by Outcome", x = "Glucose", fill = "Diabetes")
dev.off()

postscript("insulin_distribution.eps", height = 6, width = 6, paper = "special", horizontal = FALSE)
ggplot(data, aes(x = Insulin, fill = factor(Outcome))) +
  geom_histogram(position = "dodge", binwidth = 20) +
  theme_minimal() +
  labs(title = "Insulin Distribution by Outcome", x = "Insulin", fill = "Diabetes")
dev.off()

postscript("blood_pressure_distribution.eps", height = 6, width = 6, paper = "special", horizontal = FALSE)
ggplot(data, aes(x = BloodPressure, fill = factor(Outcome))) +
  geom_histogram(position = "dodge", binwidth = 5) +
  theme_minimal() +
  labs(title = "Blood Pressure Distribution by Outcome", x = "Blood Pressure", fill = "Diabetes")
dev.off()

# Statistical Tests
t_test_glucose <- t.test(Glucose ~ Outcome, data = data)
cat("T-test p-value (Glucose):", t_test_glucose$p.value, "\n")

data$HighBP <- ifelse(data$BloodPressure >= 130, 1, 0)
chi_test <- chisq.test(table(data$HighBP, data$Outcome))
cat("Chi-square p-value (High BP):", chi_test$p.value, "\n")

data$Obese <- ifelse(data$BMI >= 30, 1, 0)
prop_test <- prop.test(
  x = c(table(data$Obese, data$Outcome)[2, 1], table(data$Obese, data$Outcome)[2, 2]),
  n = c(sum(data$Outcome == 0), sum(data$Outcome == 1))
)
cat("Z-test p-value (Obesity):", prop_test$p.value, "\n")

wilcox_test <- wilcox.test(Insulin ~ Outcome, data = data)
cat("Wilcoxon p-value (Insulin):", wilcox_test$p.value, "\n")

# Logistic regression full model
set.seed(123)
index <- createDataPartition(data$Outcome, p = 0.8, list = FALSE)
train <- data[index, ]
test <- data[-index, ]

model <- glm(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness +
               Insulin + BMI + DiabetesPedigreeFunction + Age,
             data = train, family = "binomial")

# Predict on test set
pred_probs <- predict(model, test, type = "response")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
conf_matrix <- confusionMatrix(factor(pred_class), factor(test$Outcome))
print("=== Confusion Matrix ===")
print(conf_matrix)

# Risk ratio for obesity
obesity_table <- table(data$Obese, data$Outcome)
epi.2by2(obesity_table, method = "cohort.count")

# Calculate means and SDs using training data
glucose_mean <- mean(train$Glucose)
glucose_sd <- sd(train$Glucose)
insulin_mean <- mean(train$Insulin)
insulin_sd <- sd(train$Insulin)
bmi_mean <- mean(train$BMI)
bmi_sd <- sd(train$BMI)

# Standardize training data
train_scaled <- train %>%
  mutate(
    Glucose_z = (Glucose - glucose_mean) / glucose_sd,
    Insulin_z = (Insulin - insulin_mean) / insulin_sd,
    BMI_z = (BMI - bmi_mean) / bmi_sd
  )

# Standardize test data
test_scaled <- test %>%
  mutate(
    Glucose_z = (Glucose - glucose_mean) / glucose_sd,
    Insulin_z = (Insulin - insulin_mean) / insulin_sd,
    BMI_z = (BMI - bmi_mean) / bmi_sd
  )

# Fit logistic regression model with standardized variables
simple_model <- glm(Outcome ~ Glucose_z + Insulin_z + BMI_z,
                    data = train_scaled, family = "binomial")

# Show standardized coefficients
cat("Standardized Coefficients\n")
print(summary(simple_model)$coefficients)

# Predict on test set
reduced_probs <- predict(simple_model, newdata = test_scaled, type = "response")
reduced_pred <- ifelse(reduced_probs > 0.5, 1, 0)

# Confusion matrix and accuracy
conf_matrix_reduced <- confusionMatrix(factor(reduced_pred), factor(test_scaled$Outcome))
cat("Confusion Matrix: Reduced Model\n")
print(conf_matrix_reduced)

# Predict probabilities and 95% confidence intervals for new patients
new_data_raw <- data.frame(
  Glucose = c(95, 180, 140),
  Insulin = c(50, 200, 90),
  BMI = c(24.5, 37.8, 29.0)
)

# Standardize new patient data
new_data_scaled <- data.frame(
  Glucose_z = (new_data_raw$Glucose - glucose_mean) / glucose_sd,
  Insulin_z = (new_data_raw$Insulin - insulin_mean) / insulin_sd,
  BMI_z = (new_data_raw$BMI - bmi_mean) / bmi_sd
)

# Predict probabilities and confidence intervals
new_probs <- predict(simple_model, newdata = new_data_scaled, type = "link", se.fit = TRUE)
predicted_prob <- plogis(new_probs$fit)
lower <- plogis(new_probs$fit - 1.96 * new_probs$se.fit)
upper <- plogis(new_probs$fit + 1.96 * new_probs$se.fit)

cat("=== Enhanced Predictions Using Glucose, Insulin, BMI ===\n")
predictions <- data.frame(
  Glucose = new_data_raw$Glucose,
  Insulin = new_data_raw$Insulin,
  BMI = new_data_raw$BMI,
  Predicted_Probability = predicted_prob,
  CI_Lower = lower,
  CI_Upper = upper
)
print(predictions)

