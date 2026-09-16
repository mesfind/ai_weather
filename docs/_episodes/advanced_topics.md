---
title: Advanced Topics in ML
teaching: 1
exercises: 0
questions:
- ""
objectives:
- ""
keypoints:
- ""
---

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


## Crop Type Classification in Africa 

In this section, we aim to classify farm-level crop types in Kenya by leveraging Geospatial machine learning techniques with Sentinel-2 satellite imagery. This endeavor involves a supervised multiclass classification problem where we utilize pixel-level Sentinel-2 satellite imagery as the primary input for our model. The imagery comprises 12 bands of observations from Sentinel-2 L2A, encompassing various spectra such as ultra-blue, blue, green, red, visible and near-infrared (VNIR), and short wave infrared (SWIR), along with a cloud probability layer. Each pixel contains measurements for 13 dates spanning the entire farming season.

### Sentinel-2 Satellite Imagery Bands Details

The twelve bands utilized in the imagery are as follows:
- B01 (Coastal aerosol)
- B02 (Blue)
- B03 (Green)
- B04 (Red)
- B05 (Red Edge 1)
- B06 (Red Edge 2)
- B07 (Red Edge 3)
- B08 (NIR - Near Infrared)
- B8A (Red Edge 4)
- B09 (Water vapor)
- B11 (SWIR - Shortwave Infrared 1)
- B12 (SWIR - Shortwave Infrared 2)

The cloud probability layer, derived from the Sentinel-2 atmospheric correction algorithm (Sen2Cor), furnishes an estimated cloud probability (ranging from 0 to 100%) for each pixel.

### Crop Classification Categories

The objective is to classify each farm into one of the following categories:
1. Maize
2. Cassava
3. Common Bean
4. Maize & Common Bean (intercropping)
5. Maize & Cassava (intercropping)
6. Maize & Soybean (intercropping)
7. Cassava & Common Bean (intercropping)

### Model Validation and Performance Measurement

__Validation Method:__ 
We will perform a random train-validation split based on farm IDs.

__Performance Metric:__ 
The evaluation metric employed is cross-entropy. For each farm field ID, the model is expected to predict the probability of the farm containing a specific crop type.

### Data Preparation

In this phase, the dataset will undergo the following steps:
- Removal of pixels with cloud probability exceeding 50%
- Data split into train/validation/test sets
- Ensuring absence of data leakage in the sets
- Examination of channel or band distributions
- Mapping of farms by their labels
- Visualization of a single farm's NDVI evolution over time (13 dates)

~~~
import pandas as pd

df = pd.read_feather("data/df.feather")
df.head()
~~~
{: .python}

~~~
band       time       lat        lon  field  crop     B01     B02     B03     B04     B05     B06     B07     B08     B09     B11     B12     B8A
0    2019-06-06  0.168064  34.042872   2067     0  0.0192  0.0397  0.0722  0.0520  0.1063  0.2664  0.3255  0.3292  0.1973  0.1044  0.3425  0.3312
1    2019-06-06  0.168064  34.042962   2067     0  0.0192  0.0402  0.0700  0.0468  0.1085  0.3042  0.3770  0.3372  0.2088  0.1107  0.3929  0.3312
2    2019-06-06  0.168064  34.043411   1020     2  0.0248  0.0286  0.0673  0.0395  0.1144  0.3462  0.4254  0.4288  0.2325  0.1281  0.4477  0.3812
3    2019-06-06  0.168064  34.043501   1020     2  0.0248  0.0268  0.0583  0.0321  0.1085  0.3404  0.4207  0.4640  0.2368  0.1403  0.4400  0.3812
4    2019-06-06  0.168064  34.043591   1020     2  0.0248  0.0256  0.0559  0.0368  0.1085  0.3404  0.4207  0.4540  0.2368  0.1403  0.4400  0.3812
~~~
{: .output}

To visualize the fields in Kenya on a map using Sentinel-2 satellite imagery data :
~~~
# Sample one pixel per field to simplify visualization
report = df.copy()
report = report.sample(frac=1)
report = report[["field", "lat", "lon"]].drop_duplicates()
report = gpd.GeoDataFrame(report, geometry=[Point(xy) for xy in zip(report['lon'], report['lat'])])
report = report[["field", "geometry"]].drop_duplicates(subset="field")

# Get Kenya
with warnings.catch_warnings():
  warnings.filterwarnings("ignore", category=FutureWarning)
  world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
kenya = world[world['name'] == 'Kenya']

# Plot Kenya and our fields
fig, ax = plt.subplots()
_ = kenya.boundary.plot(ax=ax, color="blue")
_ = report.plot(ax=ax, color="red")
ax.set_title("Kenya (blue). Fields (red)")
plt.show()
~~~
{: .python}

![](../fig/kenya_crop_field_red.png)

To plot the data globally, showcasing country boundaries and the fields on a map as:
~~~
#@title plot the data globally

# Plot country boundaries and our fields
fig, ax = plt.subplots(figsize=(10, 7))
_ = world.boundary.plot(ax=ax, color="blue")
_ = report.plot(ax=ax, color="red")
ax.set_title("Fields (Red)")
plt.show()
~~~
{: .python}

![](../fig/kenya_crop_field_global.png)

Each (pixel, time) is a row. Let's start by removing the pixels that are cloudy:

~~~
# Drop pixels that have a cloud cover greater than 50
df = df[df["CLD"] < 50]

# No need to keep the `CLD` column anymore
df = df.drop(columns=["CLD"])
~~~

check if we have any missing values:

~~~
# Check for missing values
if df.isnull().values.any():
    # Drop rows with missing values
    df = df.dropna()
    print(f"Dropped {df.shape[0]} rows with missing values.")
else:
    print("No missing values found in the dataframe.")
~~~
{: .python}

~~~
No missing values found in the dataframe.
~~~
{: .output}


Let's  focuse on splitting the data into training, validation, and test sets based on the 'field' column. The dataset is divided into 'deploy' and the remaining rows, with 'deploy' containing hidden labels marked with 'crop == 0'. The unique field IDs are extracted from the non-deploy rows, and a random 80/10/10 split is performed for training, validation, and testing sets.

~~~
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Split data into train, val, and test sets based on field column
deploy = df[df["crop"] == 0]
train_val_test = df[~df["field"].isin(deploy["field"])]

# Randomly assign field IDs for train, val, and test sets
train_val_test_field_ids = train_val_test["field"].unique()
np.random.shuffle(train_val_test_field_ids)

# Calculate sizes for train, val, and test sets
total_fields = len(train_val_test_field_ids)
val_size = int(total_fields * 0.1)
test_size = int(total_fields * 0.1)
train_size = total_fields - val_size - test_size

# Assign field IDs to train, val, and test sets
train_field_ids = train_val_test_field_ids[:train_size]
val_field_ids = train_val_test_field_ids[train_size:train_size + val_size]
test_field_ids = train_val_test_field_ids[train_size + val_size:]

# Create train, val, and test sets
train = train_val_test[train_val_test["field"].isin(train_field_ids)]
val = train_val_test[train_val_test["field"].isin(val_field_ids)]
test = train_val_test[train_val_test["field"].isin(test_field_ids)]

# Print shapes of train, val, and test sets
train.shape, val.shape, test.shape
~~~
{: .python}

The resulting sets are named 'train', 'val', and 'test', respectively. 
~~~
((475014, 17), (59183, 17), (64530, 17))
~~~
{: .output}

Let's verify that no data leakage is happening. We define leakage as follows:

>> A validation or test farm pixels in the training dataframe (or the reverse).

~~~
# Verify that the sets of field IDs from `train`, `val`, and `test` are mutually exclusive
assert len(set(train["field"].unique()).intersection(set(val["field"].unique()))) == 0
assert len(set(train["field"].unique()).intersection(set(test["field"].unique()))) == 0
assert len(set(val["field"].unique()).intersection(set(test["field"].unique()))) == 0
~~~
{: .python}


Next, let's check the distribution of the band values we have:

~~~
import matplotlib.pyplot as plt
import seaborn as sns
g = sns.displot(data=train.drop(columns=["time", "lat", "lon", "field", "crop"]).melt().sample(100_000), x='value', hue="band", multiple='stack')
plt.title('Distribution of Input for each Band')
plt.show()
~~~
{: .python}


![](../fig/kenya_crop_band_distribution.png)

## ML Modeling

In this section, we aim to train a `LightGBM` model to predict each farm's crop type by summarizing the historical band information. We will go over the following:

- Establishing the validation metric of a frequency based model that always predicts crop type frequencies derived from y_train.
- Feature engineering: we will calculate the following S2-based indidces:

![](../fig/lighGDM_ml_modeling.png)

\\[ \begin{align*}
\text{NDVI}    &= \frac{B08 - B04}{B08 + B04} \\
\text{RDNDVI1} &= \frac{B08 - B05}{B08 + B05} \\
\text{RDNDVI2} &= \frac{B08 - B06}{B08 + B06} \\
\text{GCVI}    &= \frac{B08}{B03} - 1 \\
\text{RDGCVI1} &= \frac{B08}{B05} - 1 \\
\text{RDGCVI2} &= \frac{B08}{B06} - 1 \\
\text{MTCI}    &= \frac{B08 - B05}{B05 - B04} \\
\text{MTCI2}   &= \frac{B06 - B05}{B05 - B04} \\
\text{REIP}    &= 700 + 40 \left( \frac{(B04 + B07)/2 - B05}{B07 - B05} \right) \\
\text{NBR1}    &= \frac{B08 - B11}{B08 + B11} \\
\text{NBR2}    &= \frac{B08 - B12}{B08 + B12} \\
\text{NDTI}    &= \frac{B11 - B12}{B11 + B12} \\
\text{CRC}     &= \frac{B11 - B03}{B11 + B03} \\
\text{STI}     &= \frac{B11}{B12}
\end{align*}
\begin{equation} \\]


- Spatial median-aggregation by field ID and time.
- Conduct period-based temporal aggregation and for each band and index, create period-based columns using the following temporal groups:

- period 1
    - 2019-06-06
- period 2
    - 2019-07-01
    - 2019-07-06
    - 2019-07-11
    - 2019-07-21
- period 3
    - 2019-08-05
    - 2019-08-15
    - 2019-08-25
- period 4
    - 2019-09-09
    - 2019-09-19
    - 2019-09-24
    - 2019-10-04
- period 5
    - 2019-11-03

## Frequency-Based Baseline Model

The frequency-based baseline model is a simple approach to establish a performance reference point in machine learning. It involves:

1. **Computing class frequencies** in the training data
2. **Using these frequencies for prediction** on the validation set

This baseline provides a rudimentary performance measure that advanced models should aim to surpass, ensuring genuine value addition.


~~~
def prepare_Xy(df):
    # Check if the required columns are present
    required_columns = {'field', 'time', 'crop'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing columns in DataFrame: {missing_columns}")
    
    d = df.copy()
    
    # Remove 'band' column
    d.drop('band', axis=1, inplace=True)
    
    # Select only numeric columns for aggregation
    numeric_columns = d.select_dtypes(include='number').columns.tolist()
    
    # Group by 'field' and 'time', calculating the mean for numeric columns
    d_grouped_time = d.groupby(["field", "time"], as_index=False)[numeric_columns].mean()

    
    # Group by 'field' after dropping 'time'
    d_grouped_field = d_grouped_time.drop("time", axis=1).groupby("field", as_index=False).mean()
    
    # Separate features and target variable
    X = d_grouped_field.drop(['field', 'crop'], axis=1)
    y = d_grouped_field['crop']
    return X, y

# Prepare training and validation sets with added debug information
try:
    X_train, y_train = prepare_Xy(train)
    X_val, y_val = prepare_Xy(val)
except KeyError as e:
    print(e)

# Output shapes of the prepared data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

~~~
{: .python}

~~~
X_train shape: (2, 14)
y_train shape: (2,)
X_val shape: (2, 14)
y_val shape: (2,)
~~~
{፡ .output}

~~~
# Calculate the class frequencies from `y_train` in order to generate the baseline predictions
y_val_hat = np.repeat(y_train.value_counts(normalize=True).sort_index().values[None,...], y_val.shape[0], axis=0)
y_val_hat.shape
~~~
{: .python}

~~~
(2, 2)
~~~
{: .output

Calculate the cross entropy loss as

~~~
#@title Answer to Exercise 4 (Try not to peek until you've given it a good try!')
from sklearn.metrics import log_loss

# Calculate cross-entropy
cross_entropy = log_loss(y_val, y_val_hat)

print(f'Cross-entropy is {cross_entropy}')
~~~
{: .python}

~~~
Cross-entropy is 0.6931471805599453
~~~
{: .output}

Any model that we construct should have a validation cross-entropy less than the baseline cross-entropy.

~~~
baseline_ce = cross_entropy
baseline_ce
~~~
{: .python}

We will create functions that cover the data preparation steps in the original section description.

Let's implement the feature engineering function that would add additional vegetation indices of interest:

~~~
# @title Define feature engineering function `calculate_indices(df)` (Run Cell)
def calculate_indices(df):
    """
    Compute various spectral indices commonly used in remote sensing for vegetation monitoring.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns for the different band values.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with added columns for the calculated indices.
    """
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()

    # Normalized Difference Vegetation Index (NDVI)
    df['NDVI'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'])

    # Red-edge Normalized Difference Vegetation Index (RDNDVI)
    df['RDNDVI1'] = (df['B08'] - df['B05']) / (df['B08'] + df['B05'])
    df['RDNDVI2'] = (df['B08'] - df['B06']) / (df['B08'] + df['B06'])

    # Green Chlorophyll Vegetation Index (GCVI)
    df['GCVI'] = df['B08'] / df['B03'] - 1

    # Red-edge GCVI
    df['RDGCVI1'] = df['B08'] / df['B05'] - 1
    df['RDGCVI2'] = df['B08'] / df['B06'] - 1

    # Meris Terrestrial Chlorophyll Index (MTCI)
    df['MTCI'] = (df['B08'] - df['B05']) / (df['B05'] - df['B04'])
    df['MTCI2'] = (df['B06'] - df['B05']) / (df['B05'] - df['B04'])

    # Red-edge Inflection Point (REIP)
    df['REIP'] = 700 + 40 * (((df['B04'] + df['B07']) / 2) - df['B05']) / (df['B07'] - df['B05'])

    # Normalized Burn Ratio (NBR)
    df['NBR1'] = (df['B08'] - df['B11']) / (df['B08'] + df['B11'])
    df['NBR2'] = (df['B08'] - df['B12']) / (df['B08'] + df['B12'])

    # Normalized Difference Tillage Index (NDTI)
    df['NDTI'] = (df['B11'] - df['B12']) / (df['B11'] + df['B12'])

    # Canopy Chlorophyll Content Index (CRC)
    df['CRC'] = (df['B11'] - df['B03']) / (df['B11'] + df['B03'])

    # Soil Tillage Index (STI)
    df['STI'] = df['B11'] / df['B12']

    return df
~~~
{: .python}

We also need function for spatial and temporal aggregation to reduce the dimensionality of the dataset:

~~~
#@title Define spatial `spatial_median_aggregation(df, bands)` and temporal `period_based_aggregation(df, bands)` aggregation function (Run cell)
def spatial_median_aggregation(df, bands):
    """
    Aggregate data by field and time, using the median of band values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'field', 'time', and band columns.
    bands : list
        List of band columns to be aggregated.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with median band values.
    """
    # Calculate median of band values for each unique 'field' and 'time'
    agg_df = df.groupby(['field', 'time'])[bands].median().reset_index()

    # Drop duplicate entries for each unique 'field' and 'time', and remove band columns
    unique_df = df.drop_duplicates(['field', 'time']).drop(bands, axis=1)

    # Merge aggregated DataFrame with unique DataFrame
    return pd.merge(agg_df, unique_df, on=['field', 'time'])


def period_based_aggregation(df, bands):
    """
    Aggregate data by field and defined time periods, using the mean of band values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'field', 'time', and band columns.
    bands : list
        List of band columns to be aggregated.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with mean band values for each time period.
    """
    # Define time periods
    periods = {
        'p1': pd.to_datetime(['2019-06-06']),
        'p2': pd.to_datetime(['2019-07-01', '2019-07-06', '2019-07-11', '2019-07-21']),
        'p3': pd.to_datetime(['2019-08-05', '2019-08-15', '2019-08-25']),
        'p4': pd.to_datetime(['2019-09-09', '2019-09-19', '2019-09-24', '2019-10-04']),
        'p5': pd.to_datetime(['2019-11-03'])
    }

    # Assign period labels based on 'time'
    for period, dates in periods.items():
        df.loc[df['time'].isin(dates), 'period'] = period

    # Calculate mean of band values for each unique 'field' and 'period'
    period_agg_df = df.groupby(['field', 'period'])[bands].mean().reset_index()

    # Drop duplicate entries for each unique 'field' and 'period', and remove band columns
    unique_df = df.drop_duplicates(['field', 'period']).drop(bands, axis=1)

    # Merge aggregated DataFrame with unique DataFrame
    return pd.merge(period_agg_df, unique_df, on=['field', 'period'])
~~~
{: .python}


Finally, we create functions to pivot the table (making periods into columns) and another function that runs the steps and splits the dataframe into X and y:

~~~
#@title Define helper functions
def pivot_dataframe(df):
    """
    Pivot the DataFrame so that each time period becomes a separate column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'field', 'period', and other columns.

    Returns
    -------
    pandas.DataFrame
        Pivoted DataFrame with each 'period' as a separate column.
    """
    return df.pivot(index=['field', 'crop', 'lat', 'lon'], columns='period').fillna(-1).reset_index()


def process_dataframe(df, bands):
    """
    Process the DataFrame by calculating indices, aggregating data, and pivoting.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'field', 'time', and band columns.
    bands : list
        List of band columns to be processed.

    Returns
    -------
    X : pandas.DataFrame
        Processed DataFrame with features for machine learning.
    y : pandas.Series
        Target labels for machine learning.
    """
    # Calculate spectral indices
    df = calculate_indices(df)

    # Aggregate data by field and time using spatial median
    df = spatial_median_aggregation(df, bands)

    # Aggregate data by field and time period using mean
    df = period_based_aggregation(df, bands)

    # Calculate average latitude and longitude for each field
    lat_lon_agg = df.groupby('field')[['lat', 'lon']].mean().reset_index()

    # Merge aggregated DataFrame with latitude and longitude DataFrame
    df = pd.merge(df.drop(columns=['lat', 'lon']), lat_lon_agg, on='field', how='left')

    # Pivot DataFrame to have each period as a separate column
    df = pivot_dataframe(df)

    # Flatten multi-level column names
    df.columns = [''.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

    # Select columns to keep
    columns_to_keep = ['field', 'lat', 'lon', 'crop'] + [col for col in df.columns if col.endswith(('p1', 'p2', 'p3', 'p4', 'p5')) and not col.startswith(('time', 'lat', 'lon', 'crop'))]
    df = df[columns_to_keep]

    # Split DataFrame into features (X) and target labels (y)
    X, y = df.drop(["crop"], axis=1), df["crop"]

    return X, y
~~~
{: .python}


Let's prepare the training, validation, and test arrays:

~~~
# Set the band columns'
bands = ['B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'NDVI', 'RDNDVI1', 'RDNDVI2', 'GCVI', 'RDGCVI1', 'RDGCVI2', 'MTCI', 'MTCI2', 'REIP', 'NBR1', 'NBR2', 'NDTI', 'CRC', 'STI']

# Prepare the dataset
print("Processing `train` ...")
X_train, y_train = process_dataframe(train, bands)

print("Processing `val` ...")
X_val, y_val = process_dataframe(val, bands)

print("Processing `test` ...")
X_test, y_test = process_dataframe(test, bands)

# Print the shapes
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
~~~
{: .python}

~~~
processing `train` ...
Processing `val` ...
Processing `test` ...
((0, 3), (0,), (0, 3), (0,), (328, 133), (328,))
~~~
{: .output}

Now, let's conduct random hyperparameter search with cross-validation using the LightGBM estimator:

~~~
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import lightgbm as lgb

# Define the LightGBM model
model = lgb.LGBMClassifier(objective="multiclass", verbose=-1, num_class=7)
banned_cols = ["field"]

# Define the hyperparameters space
param_dist = {
    'num_leaves': [31, 127, 200, 300],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
}

# Define the scorer
scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

# Randomized Search for hyperparameter tuning
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=15, scoring=scorer, cv=3, verbose=1, n_jobs=-1)
random_search.fit(X_train.drop(banned_cols, axis=1), y_train)
~~~
{: .python}

### Evaluation

We re-train the best estimator on the training data and get the validation cross-entropy:

~~~
# Create the LightGBM model instance with the best hyperparameters
model = lgb.LGBMClassifier(objective="multiclass", num_class=7, verbose=-1, **random_search.best_params_)

# Fit the model to the training set
model.fit(X_train.drop(banned_cols, axis=1), y_train)

# Predict the validation set results
y_val_hat = model.predict_proba(X_val.drop(banned_cols, axis=1))

# Report cross-entropy
print(f"Cross-entropy with best hyperparameters is {log_loss(y_val, y_val_hat):.5f}")
print(f"It is {100*(log_loss(y_val, y_val_hat) - baseline_ce)/baseline_ce:.2f}% better than the baseline")
~~~
{: .python}

### Investigating Class-Imbalances

Let's report the following metrics on the combination of validation + test points:

- Precision
- Recall
- F1
- Confusion matrix

~~~
# Predict the validation set results
y_test_hat = model.predict(pd.concat([X_val, X_test]).drop(banned_cols, axis=1))
y_test_arr = pd.concat([y_val, y_test]).values
y_test_hat.shape, y_test_arr.shape
~~~
{: .python}

Calculate precision, recall, and F1 score

~~~
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# @title Answer to Exercise 5 (Try not to run until you've given it a good try!')
# Calculate precision, recall, and F1 score
precision = precision_score(y_test_arr, y_test_hat, average='weighted')
recall = recall_score(y_test_arr, y_test_hat, average='weighted')
f1 = f1_score(y_test_arr, y_test_hat, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
~~~
{: .python}

~~~
# Calculate confusion matrix
cm = confusion_matrix(y_test_arr, y_test_hat, normalize="true")

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, xticklabels=id_to_name.values(), yticklabels=id_to_name.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
~~~
{: .python}

### XAI

What are the most important periods and indices?

~~~
import shap

# Prepare the validation + test data for the model
X_vt = pd.concat([X_val, X_test])

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_vt.drop(banned_cols, axis=1))

shap.summary_plot(shap_values, X_vt.drop(banned_cols, axis=1))
~~~
{: .python}

Let's figure out which periods are the most important:

~~~
# Compute the absolute SHAP values for each feature
abs_shap_values = np.sum(np.abs(shap_values), axis=(0, 1))

# Get the feature names
feature_names = X_vt.drop(banned_cols, axis=1).columns

# Create a DataFrame linking feature names to their importance
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': abs_shap_values
})

# Sort the DataFrame by importance in descending order
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Drop `lat` and `lon` from the dataset
feature_importances = feature_importances[feature_importances["feature"].isin(["lat", "lon"]) == False]

# Normalize the feature importances to sum to one
feature_importances['importance'] = feature_importances['importance'] / feature_importances['importance'].sum()

# Split the feature name into `index` and `period` (period is the last two characters)
feature_importances['period'] = feature_importances['feature'].str[-2:]
feature_importances['index'] = feature_importances['feature'].str[:-2]
feature_importances = feature_importances.drop("feature", axis=1)

# Get the most important periods separately by aggregating their importance
periods = feature_importances.drop("index", axis=1).groupby('period').sum().sort_values('importance', ascending=False)

# Get the most important bands separately by aggregating their importance
bands = feature_importances.drop("period", axis=1).groupby('index').sum().sort_values('importance', ascending=False)
print(periods)
~~~
{: .python}

### Inference

In this section, we will report the final metrics on the validation set and visualize the farms with their crop types:

~~~
# Predict on the test set
y_test_pred = model.predict_proba(X_test.drop(banned_cols, axis=1))
y_test_pred.shape
~~~
{: .python}

~~~
# Export the results
report = X_test[["field"]].copy()

# Create the Crop_ID_1,Crop_ID_2,Crop_ID_3,Crop_ID_4,Crop_ID_5,Crop_ID_6,Crop_ID_7 columns and assign the predictions
cols = ['Crop_ID_1','Crop_ID_2','Crop_ID_3','Crop_ID_4','Crop_ID_5','Crop_ID_6','Crop_ID_7']
report[cols] = y_test_pred
report
~~~
{: .python}

~~~
from shapely.geometry import Point, LineString, Polygon

def create_geometry(df):
    coords = list(zip(df.lon, df.lat))
    if len(coords) == 1: return Point(coords[0])
    elif len(coords) == 2: return LineString(coords)
    else: return Polygon(coords)

# Create the polygons from the test set
d = test.copy()
cols = ["field", "lat", "lon"]
d = d[cols].drop_duplicates()
d = d.groupby('field').apply(create_geometry).reset_index().rename(columns={0: "geometry"})

# Create the dataframe to hold the pixel locations and the predicted crop types
report = X_test.copy()
report = report[["field"]]
report["crop"] = y_test_pred.argmax(axis=1) + 1

# Merge the two dataframes
report = report.merge(d, on="field", how="left").rename(columns={0: "geometry"})
report = gpd.GeoDataFrame(report, geometry="geometry")

# Replace the 'crop' column with mapped names
report['crop'] = report['crop'].map(id_to_name)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the GeoDataFrame using the 'crop' column to color the polygons
report.plot(column="crop", legend=True, ax=ax, cmap="Accent")

# Add a basemap
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Stamen.Terrain)

# Show the plot
plt.show()
~~~
{: .python}


## Water Table Depth Prediction


