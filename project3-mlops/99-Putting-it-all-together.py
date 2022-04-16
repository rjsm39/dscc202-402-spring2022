# Databricks notebook source
# MAGIC %md
# MAGIC # Putting it all together: Managing the Machine Learning Lifecycle
# MAGIC 
# MAGIC Create a workflow that includes pre-processing logic, the optimal ML algorithm and hyperparameters, and post-processing logic.
# MAGIC 
# MAGIC ## Instructions
# MAGIC 
# MAGIC In this course, we've primarily used Random Forest in `sklearn` to model the Airbnb dataset.  In this exercise, perform the following tasks:
# MAGIC <br><br>
# MAGIC 0. Create custom pre-processing logic to featurize the data
# MAGIC 0. Try a number of different algorithms and hyperparameters.  Choose the most performant solution
# MAGIC 0. Create related post-processing logic
# MAGIC 0. Package the results and execute it as its own run
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the<br/>
# MAGIC start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# Adust our working directory from what DBFS sees to what python actually sees
working_path = workingDir.replace("dbfs:", "/dbfs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-processing
# MAGIC 
# MAGIC Take a look at the dataset and notice that there are plenty of strings and `NaN` values present. Our end goal is to train a sklearn regression model to predict the price of an airbnb listing.
# MAGIC 
# MAGIC 
# MAGIC Before we can start training, we need to pre-process our data to be compatible with sklearn models by making all features purely numerical. 

# COMMAND ----------

import pandas as pd

airbnbDF = spark.read.parquet("/mnt/training/airbnb/sf-listings/sf-listings-correct-types.parquet").toPandas()

display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cells we will walk you through the most basic pre-processing step necessary. Feel free to add additional steps afterwards to improve your model performance.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC First, convert the `price` from a string to a float since the regression model will be predicting numerical values.

# COMMAND ----------

# Remove $ and convert to float
airbnbDF['price'] = airbnbDF['price'].replace('[\$,]', '', regex=True).astype(float)
display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at our remaining columns with strings (or numbers) and decide if you would like to keep them as features or not.
# MAGIC 
# MAGIC Remove the features you decide not to keep.

# COMMAND ----------

# Create airbnbDF with numerical columns
num_cols = airbnbDF.select_dtypes(include='number').columns.tolist()
airbnbDF_num = airbnbDF[num_cols].copy(deep=True)

# COMMAND ----------

airbnbDF.isnull().sum()

# COMMAND ----------

airbnbDF_num.nunique()

# COMMAND ----------

airbnbDF_num.describe()

# COMMAND ----------

cat_cols = airbnbDF.select_dtypes(include='object').columns.tolist()
for cat_col in cat_cols:
    print(pd.DataFrame(airbnbDF[cat_col].value_counts()))

# COMMAND ----------

# Drop the zipcode
airbnbDF1 = airbnbDF.drop(columns=['zipcode']).copy(deep=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the string columns that you've decided to keep, pick a numerical encoding for the string columns. Don't forget to deal with the `NaN` entries in those columns first.

# COMMAND ----------

# Encode the categorical columns
import numpy as np

airbnbDF1['host_is_superhost'] = np.where(airbnbDF1['host_is_superhost'] == 't', 1, 0)
airbnbDF1['instant_bookable'] = np.where(airbnbDF1['instant_bookable'] == 't', 1, 0)

# Since cancellation policy is ordinal we can label encode from low (0 - flexible) to high (4 - super_strict_60)
cancellation_policy_encoding_scheme = {"flexible": 0, "moderate": 1, "strict": 2, "super_strict_30": 3, "super_strict_60": 4}
airbnbDF1['cancellation_policy'] = airbnbDF1['cancellation_policy'].replace(cancellation_policy_encoding_scheme)

airbnbDF1 = pd.get_dummies(airbnbDF1, columns=["neighbourhood_cleansed", "property_type", "bed_type", "room_type"], prefix=["neighbourhood", "property", "bed", "room"], dtype=np.int64)

display(airbnbDF1)

# COMMAND ----------

# MAGIC %md
# MAGIC Before we create a train test split, check that all your columns are numerical. Remember to drop the original string columns after creating numerical representations of them.
# MAGIC 
# MAGIC Make sure to drop the price column from the training data when doing the train test split.

# COMMAND ----------

# Verifying there is no object/categorical dtype column
print(airbnbDF1.dtypes.value_counts())

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split

X, y = airbnbDF1.drop(columns=['price'], axis=1).copy(deep=True), airbnbDF1['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC 
# MAGIC After cleaning our data, we can start creating our model!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Firstly, if there are still `NaN`'s in your data, you may want to impute these values instead of dropping those entries entirely. Make sure that any further processing/imputing steps after the train test split is part of a model/pipeline that can be saved.
# MAGIC 
# MAGIC In the following cell, create and fit a single sklearn model.

# COMMAND ----------

# Print the names of columns missing values 
airbnbDF1.columns[airbnbDF1.isnull().any()]

# COMMAND ----------

# TODO

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Imputing with median host_total_listings_count, bathroom, beds with median 
# Imputing with median for review_scores columns
preprocessor = ColumnTransformer(
    transformers=[
        ("si1", SimpleImputer(strategy="median"), ['host_total_listings_count', 'bathrooms', 'beds']),
        ("si2", SimpleImputer(strategy="median"), ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']),
    ]
)

params = {'n_estimators': 200, 
          'max_depth': 30,
          'max_features': 'auto',
          'random_state': 202,
          'bootstrap': True
         }

clf = Pipeline(
    steps=[("Preprocessor", preprocessor), ("RFR", RandomForestRegressor(**params))]
)

clf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Pick and calculate a regression metric for evaluating your model.

# COMMAND ----------

# TODO
from sklearn.metrics import mean_squared_error

predictions = clf.predict(X_test)    

mse = mean_squared_error(y_test, predictions)
print(f"mse: {mse}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Log your model on MLflow with the same metric you calculated above so we can compare all the different models you have tried! Make sure to also log any hyperparameters that you plan on tuning!

# COMMAND ----------

# TODO
import mlflow.sklearn

with mlflow.start_run(run_name = "Run 01") as run:
    
    # Log model with name
    mlflow.sklearn.log_model(clf, "rf_model_1")
    
    # Log params
    rfr_params = clf.get_params()['RFR'].get_params()
    best_params = {x: rfr_params[x] for x in params}
    mlflow.log_params(best_params)
    
    # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mlflow.log_metric("mse", mse)
    
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Change and re-run the above 3 code cells to log different models and/or models with different hyperparameters until you are satisfied with the performance of at least 1 of them.

# COMMAND ----------

# To make our job easy 
# we will define a function to do grid serach of hyper parameters similar to previous labs

def log_best_model_grid_search(experimentID, run_name, params, X_train, X_test, y_train, y_test):
    
    from sklearn.model_selection import GridSearchCV

    with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
        
        clf = Pipeline(
            steps=[("Preprocessor", preprocessor), ("RFR", RandomForestRegressor())]
        )

        # Modify the parameter names so that they apply only to the 'Random Forest' step in the pipeline
        parameters_grid = {'RFR__'+str(key): item for key, item in params.items()}
        
        # Define Grid Search CV with param grid and 5 fold CV and n_jobs=-1 for using all processors.
        clf_gridcv = GridSearchCV(clf, parameters_grid, cv=5, n_jobs=-1)
        
        # Fit on Train set
        clf_gridcv.fit(X_train, y_train)
        
        # Get best estimator
        best_clf = clf_gridcv.best_estimator_
    
        # Create model, train it, and create predictions
        best_clf.fit(X_train, y_train)
        
        # Get predictions on Test set
        predictions = best_clf.predict(X_test)
        
        # Calculate mse for test set predictions
        mse = mean_squared_error(y_test, predictions)

        # Log model
        mlflow.sklearn.log_model(best_clf, "rf_model_1")

        # Log params
        best_params = best_clf.get_params()['RFR'].get_params()
        params1 = {x: best_params[x] for x in params}
        print(params1)
        mlflow.log_params(params1)

        # Log metrics
        mlflow.log_metric("mse", mse)
        print(f"mse: {mse}")
        
        return run.info.run_uuid

# COMMAND ----------

# Try range of values for parameters
parameters = {'n_estimators': [100, 200, 300, 500] , 
              'max_depth': [10, 20, 30, 40, 50],
             }
log_best_model_grid_search(experimentID, "Run 02", parameters, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Look through the MLflow UI for the best model. Copy its `URI` so you can load it as a `pyfunc` model.

# COMMAND ----------

# TODO
import mlflow.pyfunc

model_uri = "runs:/d1371a676e67449bac05270ae6663bd1/rf_model_1"

model1 = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-processing
# MAGIC 
# MAGIC Our model currently gives us the predicted price per night for each Airbnb listing. Now we would like our model to tell us what the price per person would be for each listing, assuming the number of renters is equal to the `accommodates` value. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fill in the following model class to add in a post-processing step which will get us from total price per night to **price per person per night**.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out <a href="https://www.mlflow.org/docs/latest/models.html#id13" target="_blank">the MLFlow docs for help.</a>

# COMMAND ----------

# TODO

class Airbnb_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        model = self.model
        output = model.predict(model_input)/model_input['accommodates']
        return output


# COMMAND ----------

# MAGIC %md
# MAGIC Construct and save the model to the given `final_model_path`.

# COMMAND ----------

# TODO

final_model_path =  f"{working_path}/final-model"

from mlflow.exceptions import MlflowException

airbnb_model = Airbnb_Model(model=model1)

dbutils.fs.rm(final_model_path, True) # Allows you to rerun the code multiple times
mlflow.pyfunc.save_model(path=final_model_path.replace("dbfs:", "/dbfs"), python_model=airbnb_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the model in `python_function` format and apply it to our test data `X_test` to check that we are getting price per person predictions now.

# COMMAND ----------

# TODO
loaded_model = mlflow.pyfunc.load_model(final_model_path)

model_input = X_test
output = loaded_model.predict(model_input)

assert output.equals(model1.predict(X_test)/X_test['accommodates'])
print("Predictions are price per person")
print(output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Packaging your Model
# MAGIC 
# MAGIC Now we would like to package our completed model! 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC First save your testing data at `test_data_path` so we can test the packaged model.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** When using `.to_csv` make sure to set `index=False` so you don't end up with an extra index column in your saved dataframe.

# COMMAND ----------

# TODO
# save the testing data

test_data_path = f"{working_path}/test_data.csv"
X_test.to_csv(test_data_path, index=False)

prediction_path = f"{working_path}/predictions.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC First we will determine what the project script should do. Fill out the `model_predict` function to load out the trained model you just saved (at `final_model_path`) and make price per person predictions on the data at `test_data_path`. Then those predictions should be saved under `prediction_path` for the user to access later.
# MAGIC 
# MAGIC Run the cell to check that your function is behaving correctly and that you have predictions saved at `demo_prediction_path`.

# COMMAND ----------

# TODO
import click
import mlflow.pyfunc
import pandas as pd

@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)

def model_predict(final_model_path, test_data_path, prediction_path):
    
    loaded_model = mlflow.pyfunc.load_model(final_model_path)
    model_input = pd.read_csv(test_data_path)
    predictions = loaded_model.predict(model_input)
    predictions.to_csv(prediction_path, index=False)

# test model_predict function    
demo_prediction_path = f"{working_path}/predictions.csv"

from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(model_predict, ['--final_model_path', final_model_path, 
                                       '--test_data_path', test_data_path,
                                       '--prediction_path', demo_prediction_path], catch_exceptions=True)

assert result.exit_code == 0, "Code failed" # Check to see that it worked
print("Price per person predictions: ")
print(pd.read_csv(demo_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create a MLproject file and put it under our `workingDir`. Complete the parameters and command of the file.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/MLproject", 
'''
name: Capstone-Project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      final_model_path: {type: str, default: "/dbfs/user/pthulasi@ur.rochester.edu/mlflow/99_putting_it_all_together_psp/final-model"}
      test_data_path: {type: str, default: "/dbfs/user/pthulasi@ur.rochester.edu/mlflow/99_putting_it_all_together_psp/test_data.csv"}
      prediction_path: {type: str, default: "/dbfs/user/pthulasi@ur.rochester.edu/mlflow/99_putting_it_all_together_psp/predictions.csv"}
    command: "python predict.py --final_model_path {final_model_path} --test_data_path {test_data_path} --prediction_path {prediction_path}"
'''.strip(), overwrite=True)

# COMMAND ----------

print(prediction_path)

# COMMAND ----------

# MAGIC %md
# MAGIC We then create a `conda.yaml` file to list the dependencies needed to run our script.
# MAGIC 
# MAGIC For simplicity, we will ensure we use the same version as we are running in this notebook.

# COMMAND ----------

import cloudpickle, numpy, pandas, sklearn, sys

version = sys.version_info # Handles possibly conflicting Python versions

file_contents = f"""
name: Capstone
channels:
  - defaults
dependencies:
  - python={version.major}.{version.minor}.{version.micro}
  - cloudpickle={cloudpickle.__version__}
  - numpy={numpy.__version__}
  - pandas={pandas.__version__}
  - scikit-learn={sklearn.__version__}
  - pip:
    - mlflow=={mlflow.__version__}
""".strip()

dbutils.fs.put(f"{workingDir}/conda.yaml", file_contents, overwrite=True)

print(file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will put the **`predict.py`** script into our project package.
# MAGIC 
# MAGIC Complete the **`.py`** file by copying and placing the **`model_predict`** function you defined above.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/predict.py", 
'''
import click
import mlflow.pyfunc
import pandas as pd

# TODO
import click
import mlflow.pyfunc
import pandas as pd

@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)

def model_predict(final_model_path, test_data_path, prediction_path):
    
    loaded_model = mlflow.pyfunc.load_model(final_model_path)
    model_input = pd.read_csv(test_data_path)
    predictions = loaded_model.predict(model_input)
    predictions.to_csv(prediction_path, index=False)
    
if __name__ == "__main__":
    model_predict()
'''.strip(), overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's double check all the files we've created are in the `workingDir` folder. You should have at least the following 3 files:
# MAGIC * `MLproject`
# MAGIC * `conda.yaml`
# MAGIC * `predict.py`

# COMMAND ----------

display( dbutils.fs.ls(workingDir) )

# COMMAND ----------

# MAGIC %md
# MAGIC Under **`workingDir`** is your completely packaged project.
# MAGIC 
# MAGIC Run the project to use the model saved at **`final_model_path`** to predict the price per person of each Airbnb listing in **`test_data_path`** and save those predictions under **`second_prediction_path`** (defined below).

# COMMAND ----------

# TODO
second_prediction_path = f"{working_path}/predictions-2.csv"
mlflow.projects.run(working_path,
  parameters={
    "final_model_path": final_model_path,
    "test_data_path": test_data_path,
    "prediction_path": second_prediction_path
})

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to check that your model's predictions are there!

# COMMAND ----------

print("Price per person predictions: ")
print(pd.read_csv(second_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> All done!</h2>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
