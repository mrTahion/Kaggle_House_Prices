import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from catboost import CatBoostRegressor


#Data filepath 
train_file_path = 'input/train.csv'
test_file_path = 'input/test.csv'

# Read the data
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

# droping outliners
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# pull data into target (y) and predictors (X)
y_train = train.SalePrice
X_train = train.drop(['SalePrice', 'Id', 'Utilities'], axis=1).select_dtypes(exclude=['object'])

# pull testing data 
X_test = test.drop(['Id', 'Utilities'], axis=1).select_dtypes(exclude=['object'])


cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    X_train[col + '_was_missing'] = X_train[col].isnull()
    X_test[col + '_was_missing'] = X_test[col].isnull()

# Imputation
my_imputer = Imputer()
X_train_imputed = my_imputer.fit_transform(X_train)
X_test_imputed = my_imputer.transform(X_test)

X_train_final = pd.DataFrame(X_train_imputed,
 columns=X_train.columns) 
X_test_final = pd.DataFrame(X_test_imputed,
 columns=X_test.columns)

# Categorial features
cat_features = X_test.select_dtypes(include='object').columns.tolist()

X_train_final, X_test_final = X_train_final.align(X_test_final,
	join='left', axis=1)

# train val split
X_train_final, X_val, y_train, y_val = train_test_split(X_train_final,
	 y_train, test_size=0.2, random_state=142)


CatBoost_model = CatBoostRegressor(
	n_estimators=15000,
	learning_rate=0.1,
	depth=5,
	l2_leaf_reg=9,
	one_hot_max_size=23
	)

CatBoost_model.fit(
	X_train_final,
	y_train,
	cat_features=cat_features,
	use_best_model=True,
	eval_set=(X_val, y_val),
	verbose=True
	)

predicted_prices = CatBoost_model.predict(X_test_final)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)