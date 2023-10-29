# Background
Heart disease is dangerous and responsible for many deaths every year. It describes any condition that affects the heart, leading to complications, such as heart failure, heart attack, and stroke. Many risk factors contribute to the development of heart diseases, such as age, smoking, and obesity. Tests and exams are performed to diagnose the disease, which includes exercise or stress tests and electrocardiograms. Motivated by this fact, the task tried to integrate machine learning research with Public Health Dataset to assist in the identification of the most influential factors or exams that help in heart disease diagnosis. It explored the anonymized dataset of Cleveland, Hungary, Switzerland, and Long Beach V from 1988. Simultaneously, it investigated if ML models may assist doctors in the diagnosis by providing a list of patients that may need more investigation. Finally, the task decided to do a few experiments. Initially, it applied clustering analysis to investigate some patterns and identify patient groups. Finally, it built two different models, one using the decision tree classifier and another using Naïve Bayes to classify the patients that may have heart disease assisting the doctors in the analysis.


# Heart Disease Dataset
The heart disease data is from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

# Data Description
This dataset contains professional medical terminology and descriptions sourced from the UCI Machine Learning Repository and Kaggle's database. It encompasses various features and target variables, each providing essential information for medical analysis. 
<br>According to the primary data sourced from the UCI Machine Learning Repository [[1](#1)] and the Kaggle database utilized in this study, comprehensive data descriptions encompassing features and target variables have been provided.

* Age – age in years
* Sex – (1 = male, 0 = female)
* Cp – chest pain type (0 = typical angina, 1 = atypical angina, 2 =
Non-anginal, 3 = asymptomatic )
* Trestbps – resting blood pressure (in mm Hg on admission to the
hospital)
* Chol – serum cholestoral in mg/dl
* Fbs – (fasting blood sugar > 120 mg/dl) (1 = true, 0 = false)
* Restecg – resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
* Thalach – maximum heart rate achieved
* Exang – exercise induced angina (1 = yes, 0 = no)
* Oldpeak – ST depression induced by exercise relative to rest
* Slope – the slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
* Ca – number of major vessels (0-3) colored by flourosopy
* Thal – (1 = normal; 2 = fixed defect; 3 = reversable defect)
* Target – (0 = no disease, 1 = disease)

# Preprocessing and Analysis of the data
## Target Data Checking

![Alt text](images/heart_disease_target.png)

The bar chart and corresponding calculations reveal that 526 individuals exhibit the presence of heart disease, accounting for approximately 51.32% of the dataset, while 499 individuals exhibit an absence of heart disease, representing approximately 48.68% of the total. This indicates a balanced distribution within the target data.

## Analysis
### Feature relationship between variables

![Alt text](images/feature_relationship_heatmap.png)

From the heatmap analysis, the following observations can be made:

1. Variables such as Oldpeak, exang, thalach, and cp, with coefficient exceeding 0.4, exhibit a strong and significant correlation with the presence of heart disease.

2. Features including thal, ca, slope, age, and sex, with coefficient falling within the range of 0.2 to 0.4, display a moderate and notable association with heart disease.

3. Variables trestbps, chol, fbs, and restecg, characterized by coefficient below 0.2, demonstrate a relatively weak correlation with heart disease.

4. Notably, the feature Oldpeak is strongly correlated with slope, as are thalach with age, and cp with exang, revealing interesting interrelationships among these variables.

### Visualiztion
Relationship between heart disease and oldpeak

![Alt text](images/oldpeak_vs_target.png)

Relationship between oldpeak and slope

![Alt text](images/oldpeak_vs_slope.png)

Relationship between exang and heart disease

![Alt text](images/exang_vs_target.png)

Relationship between exang and cp

![Alt text](images/exang_vs_cp.png)

Relationship between cp and heart disease

![Alt text](images/cp_vs_target.png)

Relationship between thalach and age and heart disease

![Alt text](images/thalach_vs_age_vs_target.png)

Relationship between sex and heart disease

![Alt text](images/sex_vs_target.png)

Relationship between ca and heart disease

![Alt text](images/ca_vs_target.png)

relationship between thal and heart disease

![Alt text](images/thal_vs_target.png)


## Clusters Evaluation
### Feature Selection

![Alt text](images/feature_importance.png)

The dataset was partitioned into two subsets using a threshold of 10, resulting in a categorical set and a numeric set. The categorical set comprises the features 'sex,' 'cp,' 'fbs,' 'restecg,' 'exang,' 'slope,' 'ca,' 'thal,' and 'target,' while the numeric set includes 'age,' 'trestbps,' 'chol,' 'thalach,' and 'oldpeak.' Given the prevalence of categorized features, the feature selection process was executed utilizing the RandomForestClassifier model. This selection yielded five most significant features: 'cp,' 'ca,' 'oldpeak,' 'thal,' and 'thalach.' Subsequently, one-hot encoding was applied to the categorical features 'cp,' 'ca,' and 'thal.'

### K-Means 
Apply the K-means clustering method to derive patient profiles from the dataset, excluding labels. To assess the quality of the clusters, utilize the silhouette measure to choose the optimal number of clusters (k) that maximizes the silhouette score for the dataset.

![Alt text](images/silhouette_score.png)

The results indicate that as the value of "k" increases, the silhouette score exhibits a general decreasing trend. The highest silhouette score, approximately 0.57, is achieved when "k" equals 2. 

### Illustrate the clusters in a 2D dimension using PCA

![Alt text](images/pca_cluster.png)

### Rand Index
Rand Index of 0.57687 indicates that there is some level of agreement between the predicted results of the cluster and the labels given in the dataset, but it is not a perfect match. The value is greater than 0.5, which implies that there is more similarity than dissimilarity.

## Compare the models Naïve bayes (NB) and Decision tree (DT) 

The analysis reveals that the Naive Bayes model achieves an accuracy of approximately 0.81493 and the Decision Tree model exhibits comparable performance, with an accuracy of 0.80519. 

The confusion matrix illustrates 118 correct predictions of heart disease, 41 false predictions of no heart disease when the patients have heart disease, 16 false predictions of heart disease when they have no heart disease, and 133 correct predictions of no heart disease. The confusion matrix for the Decision Tree model shows 116 correct predictions of heart disease, 43 false predictions of heart disease, 17 false predictions of no disease when patients have heart disease, and 132 correct predictions of no heart disease. Evaluation metrics such as accuracy, precision, recall, and f1-score, along with the ROC curve, are used to assess the models, providing insights into their predictive capabilities. The Naive Bayes model outperforms the Decision Tree model slightly, possibly because the Naive Bayes model is less affected by feature correlations due to its independence assumption.

Additionally, McNemar's test reveals no statistically significant difference in model performance, as the p-value exceeds 0.5, failing to reject the null hypothesis. This suggests that both models perform similarly in predicting heart disease.

# References
<a id="1">[1]</a>
R. Detrano, A. Jánosi, W. Steinbrunn, M. Pfisterer, J. Schmid, at al. (1989). "Heart Disease". *Published in American Journal of Cardiology* [Online]. Available: https://archive.ics.uci.edu/dataset/45/heart+disease. Accessed: Oct. 26, 2022
