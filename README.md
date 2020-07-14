# House Price Prediction : Overview

House Prices: Advanced Regression Techniques is one of the prediction competitions in Kaggle.com. This prediction project is about predicting house prices based upon the different features of the house which the general buyers might look into. The features might be utilities avaialability, number of bedrooms, garage conditions, alley acces, conditions of basement, shape, structures, date of construction and overall conditions. In addition, this project also gives proper insight on how the different features influences the houseprices. The Ames Housing dataset was compiled by Dean De Cock for use in data science education. This project presents a data set describing the sale of individual residential property in Ames, Iowa from 2006 to 2010. 

# Code and Resources Used

* Python Version: 3.7
* Packages: pandas, numpy, sklearn, matplotlib, seaborn
** Dataset : https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# Data Description
Here's a brief version of what you'll find in the data description file.

* SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
* MSSubClass: The building class
* MSZoning: The general zoning classification
* LotFrontage: Linear feet of street connected to property
* LotArea: Lot size in square feet
* Street: Type of road access
* Alley: Type of alley access
* LotShape: General shape of property
* LandContour: Flatness of the property
* Utilities: Type of utilities available
* LotConfig: Lot configuration
* LandSlope: Slope of property
* Neighborhood: Physical locations within Ames city limits
* Condition1: Proximity to main road or railroad
* Condition2: Proximity to main road or railroad (if a second is present)
* BldgType: Type of dwelling
* HouseStyle: Style of dwelling
* OverallQual: Overall material and finish quality
* OverallCond: Overall condition rating
* YearBuilt: Original construction date
* YearRemodAdd: Remodel date
* RoofStyle: Type of roof
* RoofMatl: Roof material
* Exterior1st: Exterior covering on house
* Exterior2nd: Exterior covering on house (if more than one material)
* MasVnrType: Masonry veneer type
* MasVnrArea: Masonry veneer area in square feet
* ExterQual: Exterior material quality
* ExterCond: Present condition of the material on the exterior
* Foundation: Type of foundation
* BsmtQual: Height of the basement
* BsmtCond: General condition of the basement
* BsmtExposure: Walkout or garden level basement walls
* BsmtFinType1: Quality of basement finished area
* BsmtFinSF1: Type 1 finished square feet
* BsmtFinType2: Quality of second finished area (if present)
* BsmtFinSF2: Type 2 finished square feet
* BsmtUnfSF: Unfinished square feet of basement area
* TotalBsmtSF: Total square feet of basement area
* Heating: Type of heating
* HeatingQC: Heating quality and condition
* CentralAir: Central air conditioning
* Electrical: Electrical system
* 1stFlrSF: First Floor square feet
* 2ndFlrSF: Second floor square feet
* LowQualFinSF: Low quality finished square feet (all floors)
* GrLivArea: Above grade (ground) living area square feet
* BsmtFullBath: Basement full bathrooms
* BsmtHalfBath: Basement half bathrooms
* FullBath: Full bathrooms above grade
* HalfBath: Half baths above grade
* Bedroom: Number of bedrooms above basement level
* Kitchen: Number of kitchens
* KitchenQual: Kitchen quality
* TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
* Functional: Home functionality rating
* Fireplaces: Number of fireplaces
* FireplaceQu: Fireplace quality
* GarageType: Garage location
* GarageYrBlt: Year garage was built
* GarageFinish: Interior finish of the garage
* GarageCars: Size of garage in car capacity
* GarageArea: Size of garage in square feet
* GarageQual: Garage quality
* GarageCond: Garage condition
* PavedDrive: Paved driveway
* WoodDeckSF: Wood deck area in square feet
* OpenPorchSF: Open porch area in square feet
* EnclosedPorch: Enclosed porch area in square feet
* 3SsnPorch: Three season porch area in square feet
* ScreenPorch: Screen porch area in square feet
* PoolArea: Pool area in square feet
* PoolQC: Pool quality
* Fence: Fence quality
* MiscFeature: Miscellaneous feature not covered in other categories
* MiscVal: $Value of miscellaneous feature
* MoSold: Month Sold
* YrSold: Year Sold
* SaleType: Type of sale
* SaleCondition: Condition of sale

# Data Preprocessing

* The pre processing of data begins by looking out for missing values in each explainatory variables.
* For this dataset I am considering to eliminate those variables which have or have more than 60% of total data missing or showing no values. PoolQC, MiscFeature, Alley and Fence attributes for predicting house prices have very large numbers of missing values, which are, 0.99, 0.96, 0.93 and 0.80 respectively. Since it covers almost whole columns eliminating these attributes won't cause any significant loss of values to our project. It is also important to be aware that right evaluation is required to deal with such missing values because there may be chance of loss of important data.
* LotFrontage will be considered in our dataset. It has 259 missing values which is significant number and will replace with Median. However, I think it has moderate relationship with house price.
* Before analyzing Garage related variables, it seems all variables are related with having or not having garage in the house. Garagecars do have good relationship with the price of the house. Since our aim is the least deletion of data, we will consider all other data.
* Garage Type: This variable describes the location of garage. So, 'na' does make sense that there are no garage in the house. So, I will create new category No Garage.
* GarageYrBuilt: This variable describes the year garage was built. So, I will make this variable the categorical variable. The data with na will go into 'Unknown' category.
* Garage Finish: This categorical variable describes interior finishes of the garage. The 'na's indicating missing values do make sense here that it represents there are no garages. The data with na will go into 'No Garage' category.
* Garage Qual: This categorical variable describes quality of the garage. The 'na' indicates the houses with no garage. The na values will go into No Garage category.
* BsmtExposure: The 'na' values can be put into No Basement category.
* BsmtFinType2: The 'na' values can be put into No Basement category.
* BsmtFinType1: The 'na' values can be put into No Basement category.
* BsmtCond: It indicates condtion of the basement. The 'na' values can be put into No Basement category.
* BsmtQual: It indicates overall qualties of the basement. The 'na' values can be put into No Basement category.
* MasVnrType: It indicates what type of Masonery venner has been put into the house and it could be one of the significant variable that influences selling price. We can categorize na values into Unknown category.
* MasVnrArea: It indicates area of Masonery venner and it could be one of the significant variable that influences selling price. Since we only have 8 missing values, we will impute those values with Median value.
* Since Electrical is categorical variable, I would consider to delete one and only missing value.
* FireplaceQu is categorical variable; I will make another category for null values. 'No Fireplace' replaces the missing values of FireplaceQu. 
* Checking the columns again if there are any other missing values.

# Exploratory Data Analysis

* The plotting of the distribution curve for the SalePrice to check it is normally distributed.
* Data types are checked. It appears that all varaibles have appropriate values.
* The variables will be divided into numerical and categorical variables.
* Categorical data or qualitative data divides a variable into groups which represents data. Boxplot is used here for visualizing categorical variable. The boxplot displays how categorical variables are associated with target variable 'SalePrice'. It shows mean, median, standard deviaiton,ranges, minimum and maximum values and outliers. Generally, it displays how data are distributed, their relationship with the target variable and helps to study and understand data in meaningful way.
* Histograms for all numeric variables to determine if all variables are skewed.
* Scatter plot is used to show linear relationship between variables. It is a technique to get insights for regression analysis.
* MasVnrArea,LotArea, LotFrontage, YearBuilt , BsmtFinSF1, TotalBsmtSF, 1stFlrSF, GrLivArea and GarageArea seems to be strong contender for predicting SalePrice.
* YearRemodAdd,BsmtFinSF2,BsmtUnfSF,2ndFlrSF, LowQualFinSF, WoodDeckSF, OpenPorchSF, EnclosedPorch, ScreenPorch, PoolArea and MiscVal doesn't seem to be good predictors as the line is close to horizontal and less correlated.
* Describe method is used here to understand statistical summaries and measures of data. This will show the count of that variable, the mean, the standard deviation (std),the minimum value, the IQR (Interquartile Range: 25%, 50% and 75%) and the maximum value.
* Correlation coefficient is a measure which helps to visually judge the specific strength of a relationship. It measures the dependencies between target variable and independent variables. 
* Calculation of Correlation Coefficient and P-value of the numeric variables and the target variable, SalePrice
* Since the p-value of MasVnrArea, LotFrontage, LotArea, YearBuilt, YearRemodAdd, BsmtFinSF1, BsmtUnfSF, TotalBsmtSF,1stFlrSF, 2ndFlrSF,GrLivArea, GarageArea, WoodDeckSF,OpenPorchSF, ScreenPorch, EnclosedPorch and PoolArea are < 0.001, their correlation is statistically significant.
* Since the p-value of BsmtFinSF2, LowQualFinSF and MiscVal are > 1, there is no evidence that their correlation is significant.
* Since the p-value of 3SsnPorch is <0.1, there is weak evidence that the correlation is significant.¶
* Bar plot makes easier to compare each group of categorical variable.

# Model Building

* The categorical variables are converted into dummy variables. A dummy variable is a numerical variable used in regression analysis to represent subgroups of the variable.
* The data split into training data and test data. 
* Three different models are in use. They are Multiple Linear Regression, LASSO Regression and Random Forest.
* Cross validation is used as its score compares and selects a model for a given predictive modeling problem because it is easy to understand, easy to implement, and results in skill estimates that generally have a lower bias than other methods.
* The models are used to predict test set data.
* Based on mean absolute error, the predicted values are evaluated.
* The LASSO model performs better (MAE = 0.08) than other models on the test and validation sets.

   * Multiple Linear Regression (MAE) : 0.09938720120653549
   * LASSO Regression (MAE) : 0.08296194993733137
   * Random Forest (MAE) : 0.10594839677613957

