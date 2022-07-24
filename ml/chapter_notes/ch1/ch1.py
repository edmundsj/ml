from ch1_includes import *

# First, grab the data from the server and extract it
fetch_housing_data()

# Next, load the data into a pandas dataframe
raw_data = load_housing_data()
train_data, test_data = train_test_split(raw_data, test_size=0.2)

# Preprocess the data to split into 'labels' and data
housing_labels_train = train_data['median_house_value'].copy()
housing_labels_test = test_data['median_house_value'].copy()

train_data = train_data.drop('median_house_value', axis=1)
test_data = test_data.drop('median_house_value', axis=1)

# Remove non-numerical data temporarily
housing_numerical = train_data.drop('ocean_proximity', axis=1)

# Finally, implement the whole thing with a pipeline. This is where using classes for transformation
# rather than variables really starts to shine. You can build up a parameterizable transformation
# pipeline at the beginning and execute it all at once, with no need for intermediate variable names.
# Generally speaking this seems like a brilliant idea and I should definitely do more of it. 
# I should incorporate this model into my own data processing. Will make it much more modular and
# easy to build up complex data processing pipelines.

numerical_attributes = list(housing_numerical)
categorical_attributes = ['ocean_proximity']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),      # Replaces missing numerical data with the median
    ('attrib_adder', CombinedAttributesAdder()),        # Adds potentially useful attributes
    ('std_scaler', StandardScaler())                    # Performs feature scaling
])

full_pipeline = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_attributes),
    ('categorical', OneHotEncoder(), categorical_attributes)
])

train_prepared = full_pipeline.fit_transform(train_data)
test_prepared = full_pipeline.transform(test_data)

lin_reg = LinearRegression()
validate_and_report(lin_reg, train_prepared, housing_labels_train, name='Linear',
                    cv=10, scoring='neg_mean_squared_error')

tree_reg = DecisionTreeRegressor()
validate_and_report(tree_reg, train_prepared, housing_labels_train, name='Tree',
                    cv=10, scoring='neg_mean_squared_error')

forest_reg = RandomForestRegressor()
validate_and_report(forest_reg, train_prepared, housing_labels_train, name='Forest',
                    cv=10, scoring='neg_mean_squared_error')
param_grid = [{
        'n_estimators': [3, 30],
        'max_features': [2, 4]
    },
    {
        'bootstrap': [False],
        'max_features': [2, 4],
        'n_estimators': [3, 30]
    }
]

#grid_search = GridSearchCV(forest_reg, param_grid, cv=3,
#                           scoring='neg_mean_squared_error', return_train_score=True)
#grid_search.fit(train_prepared, housing_labels_train)
