#Load required libraries and install packages
library(factoextra)
library(cluster)
library(ggplot2)
library(readxl)
library(caret)

# Load dataset
df <- readxl::read_excel("C:/Users/Viki/Downloads/Face expression recognition (2)/Face expression recognition/cohn-kanade-rev_new.xlsx")

# Inspect dataset
str(df)
summary(df)

#counting the coloumns of the dataset
ncol(df)

#check for missing values
colSums(is.na(df))

# Remove the unnecessary columns
df <- df[, colSums(is.na(df)) == 0] 
head(df)
ncol(df)

# Remove Expression column 
df_features <- df[, !colnames(df) %in% "Expression"]
head(df)

# Normalize the features
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df_scaled <- as.data.frame(lapply(df_features, normalize))
print(df_scaled)

# Compute WCSS for different K values
set.seed(123)
fviz_nbclust(df_scaled, kmeans, method = "wss") + 
  ggtitle("Elbow Method for Optimal K")

# Choose K based on the elbow method 
set.seed(123)
k_value <- 7  
kmeans_result <- kmeans(df_scaled, centers = k_value, nstart = 25)

# Add cluster labels to dataset
df_scaled$Cluster <- as.factor(kmeans_result$cluster)

df_original <- df 
df_original$Cluster <- as.factor(kmeans_result$cluster)

# Create a confusion matrix
table(Predicted_Cluster = df_original$Cluster, Actual_Expression = df$Expression)

#Perform PCA Analysis
pca_result <- prcomp(df_scaled[, -ncol(df_scaled)], center = TRUE, scale. = TRUE)
pca_df <- data.frame(PC1 = pca_result$x[, 1], PC2 = pca_result$x[, 2], Cluster = df_scaled$Cluster)

# Plot the PCA results with clusters
ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  ggtitle("K-Means Clustering Visualization (PCA)")

# Add actual expressions to the PCA data frame
pca_df$Expression <- as.factor(df$Expression)

# Plot PCA with actual expression labels
ggplot(pca_df, aes(x = PC1, y = PC2, color = Expression, label = Expression)) +
  geom_point(size = 3) +  # Plot points
  geom_text(vjust = 1.5, hjust = 0.5, size = 3, alpha = 0.7) +  
  ggtitle("K-Means Clustering Visualization (PCA) with Emotion Labels")

# Making Expression a factor
df$Expression <- as.factor(df$Expression)

# Check for missing values
if (anyNA(df)) {
  df <- na.omit(df)
}

# Split the data
set.seed(123)
trainIndex <- createDataPartition(df$Expression, p = 0.7, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Ensure train is a data frame
train <- as.data.frame(train)
test <- as.data.frame(test)

# Set up cross-validation
control <- trainControl(method = "cv", number = 10)

# Train SVM model
svm_model <- train(Expression ~ ., data = train, method = "svmLinear", trControl = control)
print(svm_model)

# Train Random Forest model
rf_model <- train(Expression ~ ., data = train, method = "rf", trControl = control)
print(rf_model)

# Predict on test set
pred_svm <- predict(svm_model, test)
pred_rf <- predict(rf_model, test)

# Evaluate accuracy
confusionMatrix(pred_svm, test$Expression)
confusionMatrix(pred_rf, test$Expression)
