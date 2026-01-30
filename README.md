# California Housing Price Prediction – Linear Regression

Project Overview

This project builds a Linear Regression model to predict median house values in California districts using the California Housing dataset from scikit-learn.  
The goal is to go end-to-end: data loading, exploratory analysis, model training, evaluation, and interpretation of feature impact on prices. 

 Dataset

- Source: 'sklearn.datasets.fetch_california_housing'  
- Instances: 20,640 rows of district-level data.   
- Features (inputs):   
  - 'MedInc' – median income in block group  
  - 'HouseAge' – median house age  
  - 'AveRooms' – average number of rooms per household  
  - 'AveBedrms' – average number of bedrooms per household  
  - 'Population' – block group population  
  - 'AveOccup' – average household size  
  - 'Latitude' – district latitude  
  - 'Longitude' – district longitude  
- Target(output) 
  - 'MedHouseVal' – median house value (in units of 100,000 USD)


## Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn (LinearRegression, train_test_split, metrics)  
- Matplotlib (for visualization)



## Project Steps

1. Load and inspect data 
   - Loaded the California Housing dataset using 'fetch_california_housing(as_frame=True)'.  
   - Converted to a Pandas DataFrame and inspected it with `head()`, `info()`, and `describe()` to understand data types and value ranges.

2. Feature and target selection 
   - Defined X as all feature columns ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude').  
   - Defined y as 'MedHouseVal', a continuous numeric target suitable for regression. 

3. Train–test split 
   - Split the dataset into training and testing sets (80% train, 20% test) using 'train_test_split' with 'random_state=42' for reproducibility. 

4. Model training 
   - Trained a 'LinearRegression' model on the training data using scikit-learn. [web:6][web:10]

5. Prediction and comparison table 
   - Generated predictions on the test set.  
   - Created a small comparison table (first 10 rows) showing Actual vs Predicted house values to manually inspect prediction quality.

6. Model evaluation (MAE & RMSE)  
   - Computed Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the test set.   
   - Since 'MedHouseVal' is in 100,000 USD, errors can be interpreted in terms of real currency (e.g., 0.5 ≈ 50,000 USD). 

7. Visualization  
   - Plotted Actual vs Predicted values using a scatter plot.  
   - Added a 45-degree reference line to visually check how close predictions are to true values.

8. Coefficient interpretation  
   - Extracted model coefficients and mapped them to feature names.  
   - Interpreted which features have the strongest positive or negative influence on predicted house prices.


 Results

 Error metrics

- MAE: '0.5332'  
- RMSE: '0.7456'  

Because the target is in units of 100,000 USD, these values mean:  

- MAE = 0.53 → average absolute error of around 53,000 USD per prediction.   
- RMSE = 0.75 → typical error magnitude around 75,000 USD, with larger errors penalized more strongly. ]

These metrics summarize how far the model’s predictions are from true house values on unseen test data. 

### Feature coefficients (impact on price)

Learned coefficients from the linear regression model:

| Feature    | Coefficient |
|-----------|------------:|
| AveBedrms | 0.783145    |
| MedInc    | 0.448675    |
| HouseAge  | 0.009724    |
| Population| -0.000002   |
| AveOccup  | -0.003526   |
| AveRooms  | -0.123323   |
| Latitude  | -0.419792   |
| Longitude | -0.433708   |

Interpretation (holding other features constant):

- AveBedrms (0.783145): Higher average bedrooms per household are strongly associated with higher predicted house values.  
- MedInc (0.448675): Higher median income in a district is associated with higher house prices, aligning with real-world expectations.  
- HouseAge (0.009724): Older houses have a slight positive association with price, but the effect is modest.  
- AveRooms (-0.123323): More rooms per household is associated with lower prices after controlling for other features, which can reflect interactions or multicollinearity in the data.
- Latitude (-0.419792) and Longitude (-0.433708): Moving in certain geographic directions is associated with lower prices, reflecting location effects (e.g., distance from high-demand coastal areas).  
- AveOccup (-0.003526): Higher occupancy per household is slightly associated with lower prices, possibly indicating crowding.  
- Population (-0.000002): Coefficient is almost zero, suggesting little direct linear effect of population on price.



 Files in This Repository

- 'house prices.ipynb' – Full workflow: data loading, EDA, model training, evaluation, and plotting.  
- 'actual_vs_predicted.png' – Scatter plot of Actual vs Predicted house values.  
- 'actual_vs_predicted_sample.csv' – Sample comparison of Actual vs Predicted values for some test rows (optional).  
- 'README.md' – This project description and results summary.

---

## How to Run

1. Clone the repository:
   
   git clone <https://github.com/PranithaBokketi/task6-california-house-prices/new/main?filename=README.md>

2. Install dependencies:

   pip install numpy pandas scikit-learn matplotlib
   
3. Open and run the notebook:
   house prices.ipynb
4. Run all cells to reproduce the results and plots.

 Key Learning Outcomes

 Used Linear Regression to predict continuous house prices on a real-world dataset. 

Understood why train–test split is needed to evaluate generalization. 

Practiced using MAE and RMSE to measure regression performance and interpret them in real currency units. 

Interpreted linear model coefficients to understand which features most affect house prices in this dataset. 



  
