# Assignment-1-APR-CS502

## Titanic Survival Prediction 

## Project Overview
This project analyzes the Titanic dataset to predict passenger survival using machine learning algorithms, with a focus on Principal Component Analysis (PCA) for dimensionality reduction and feature analysis.

## Dataset Information
- **Source**: Titanic dataset with passenger information
- **Size**: 891 passengers, 12 original features
- **Target**: Survival prediction (0 = Did not survive, 1 = Survived)
- **Missing Data**: Age (177), Cabin (687), Embarked (2)

## Methodology

### 1. Data Preprocessing
- **Missing Value Treatment**: 
  - Age: Filled with median (28.0)
  - Embarked: Filled with mode ('S')
  - Cabin: Dropped due to 77% missing values
- **Feature Engineering**:
  - Extracted titles from passenger names
  - Created FamilySize (SibSp + Parch + 1)
  - Created IsAlone indicator (FamilySize == 1)
- **Encoding**: Label encoding for categorical variables

### 2. Principal Component Analysis (PCA)
- **Purpose**: Dimensionality reduction and feature importance analysis
- **Results**: 8 components explain 97.1% of data variance
- **Key Findings**:
  - PC1 (32.7%): Family-related features (FamilySize, IsAlone, SibSp)
  - PC2 (20.0%): Socio-economic features (Pclass, Fare, Age)
  - PC3 (12.7%): Personal attributes (Title, Embarked, Age)

### 3. Feature Importance Analysis
**Top Features by Correlation with Survival:**
1. Sex: 0.543 (strongest predictor)
2. Pclass: -0.338 (passenger class)
3. Fare: 0.257 (ticket fare)
4. IsAlone: -0.203 (traveling alone)
5. Embarked: -0.168 (port of embarkation)

### 4. Machine Learning Models
**Models Tested:**
- Logistic Regression (Original & PCA features)
- Random Forest (Original & PCA features)

**Performance Results:**
1. Random Forest (Original): **82.68% accuracy** ⭐ Best Model
2. Random Forest (PCA): 82.12% accuracy
3. Logistic Regression (Original): 81.01% accuracy
4. Logistic Regression (PCA): 81.01% accuracy

## Key Insights

### Survival Patterns
- **Overall Survival Rate**: 38.4%
- **Gender Impact**: Women: 74.2% vs Men: 18.9%
- **Class Impact**: 1st: 62.9%, 2nd: 47.3%, 3rd: 24.2%
- **Family Effect**: Passengers with family had better survival rates

### PCA Insights
- Successfully reduced 10 features to 8 components with minimal information loss
- Family relationships and socio-economic status are primary variance drivers
- PCA provides interpretable feature groupings

### Model Performance
- Random Forest outperforms Logistic Regression
- Original features slightly better than PCA-transformed features
- Feature engineering improves model performance significantly

## Technical Implementation

### Data Split
- Training: 712 samples (80%)
- Testing: 179 samples (20%)
- Stratified split to maintain class balance

### Evaluation Metrics
- Accuracy: 82.68%
- Precision (Not Survived): 85%
- Precision (Survived): 79%
- Recall (Not Survived): 87%
- Recall (Survived): 75%

## Files Generated
1. `titanic_dataset_summary.csv` - Dataset statistics
2. `titanic_processed_dataset.csv` - Cleaned data
3. `titanic_pca_analysis.csv` - PCA results
4. `titanic_feature_importance.csv` - Feature rankings
5. `titanic_model_comparison.csv` - Model performance
6. `titanic_predictions.csv` - Test predictions
7. `titanic_analysis_complete.py` - Complete code implementation

## Conclusions

### Best Approach
- **Algorithm**: Random Forest Classifier
- **Features**: Original preprocessed features (10 features)
- **Accuracy**: 82.68% on test set

### PCA Effectiveness
- PCA successfully identified key feature groups
- Minimal performance loss (0.56%) with dimensionality reduction
- Excellent for understanding data structure and visualization

### Recommendations
1. Use Random Forest for best prediction accuracy
2. Include all engineered features for optimal performance
3. PCA valuable for feature understanding and dimension reduction
4. Gender and passenger class are strongest survival predictors

## Code Implementation
Complete Python implementation available in `titanic_analysis_complete.py` with:
- Data preprocessing pipeline
- PCA analysis and visualization
- Model training and evaluation
- Result generation and export

**Assignment successfully completed with comprehensive analysis and 82.68% prediction accuracy!**