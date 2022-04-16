# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Adding Pre and Post-Processing Logic
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Import data and train a random forest model
# MAGIC  - Defining pre-processing steps
# MAGIC  - Adding post-processing steps
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

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data and Train Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC Import the Airbnb DataFrame.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Train a random forest model.

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rf = RandomForestRegressor(n_estimators=100, max_depth=25)
rf.fit(X_train, y_train)
rf_mse = mean_squared_error(y_test, rf.predict(X_test))

rf_mse

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-processing Our Data
# MAGIC 
# MAGIC We would like to add some pre-processing steps to our data before training a RF model in order to decrease the MSE and improve our model's performance.
# MAGIC 
# MAGIC Take a look at the first 10 rows of our data.

# COMMAND ----------

df.iloc[:10]

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Notice that all the values in the `latitude` and `longitude` columns are very similar (up to tenth place) since all the Airbnb listings are in San Francisco. The Airbnb pricing probably will not vary too much between longitude and latitude differences of 0.0001 so we can facilitate the splitting factors of our tree by rounding the `latitude` and `longitude` values to the nearest hundredth instead of worrying about all 6 digits after the decimal point. We will create these values in new columns called `trunc_lat` and `trunc_long` and drop the original `latitude` and `longitude` columns.
# MAGIC 
# MAGIC Additionally, notice that the 'review_scores_accuracy',
# MAGIC        'review_scores_cleanliness', 'review_scores_checkin',
# MAGIC        'review_scores_communication', 'review_scores_location', and
# MAGIC        'review_scores_value'
# MAGIC        encode pretty similar information so we will go ahead and summarize them into single column called `summed_review_scores` which contains the summation of the above 6 columns. Hopefully the tree will be able to make a more informed split given this additional information.
# MAGIC 
# MAGIC 
# MAGIC Fill in the pre-processing lines to create the `X_test_processed` and `X_train_processed` DataFrames. Then we will train a new random forest model off this pre-processed data.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Take a look at python's built in `round` function.

# COMMAND ----------

# TODO
# new random forest model
rf2 = RandomForestRegressor(n_estimators=100, max_depth=25)

# pre-process train data
X_train_processed = X_train.copy()
X_train_processed["trunc_lat"] = X_train_processed["latitude"].round(2)
X_train_processed["trunc_long"] = X_train_processed["longitude"].round(2)
X_train_processed["review_scores_sum"] = X_train_processed[['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']].sum(axis=1)
X_train_processed = X_train_processed.drop(columns=['latitude','longitude'], axis=1)

# pre-process test data to obtain MSE
X_test_processed = X_test.copy()
X_test_processed["trunc_lat"] = X_test_processed["latitude"].round(2)
X_test_processed["trunc_long"] = X_test_processed["longitude"].round(2)
X_test_processed["review_scores_sum"] = X_test_processed[['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']].sum(axis=1)
X_test_processed = X_test_processed.drop(columns=['latitude','longitude'], axis=1)


# fit and evaluate new rf model
rf2.fit(X_train_processed, y_train)
rf2_mse = mean_squared_error(y_test, rf2.predict(X_test_processed))

rf2_mse

# COMMAND ----------

# MAGIC %md
# MAGIC After training our new `rf2` model, let us log this run in MLflow so we can use this trained model in the future by loading it.

# COMMAND ----------

import mlflow.sklearn

with mlflow.start_run(run_name="RF Model Pre-process") as run: 
    
    mlflow.sklearn.log_model(rf2, "random-forest-model-preprocess")
    mlflow.log_metric("mse", rf2_mse)

    experimentID = run.info.experiment_id
    runID = run.info.run_uuid
    artifactURI = mlflow.get_artifact_uri()

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's load the `python_function` flavor of the model so we can apply it to a test set.

# COMMAND ----------

# import mlflow.pyfunc
# from  mlflow.tracking import MlflowClient

# client = MlflowClient()
# rf2_run = sorted(client.list_run_infos(experimentID), key=lambda r: r.start_time, reverse=True)[0]
# rf2_path = rf2_run.artifact_uri+"/random-forest-model-preprocess/"

import mlflow.pyfunc
rf2_pyfunc_model = mlflow.pyfunc.load_model("runs:/"+runID+"/random-forest-model-preprocess")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try giving our new `rf2_pyfunc_model` the `X_test` DataFrame to generate predictions off of.

# COMMAND ----------

try:
  rf2_pyfunc_model.predict(X_test)
except ValueError as e:
  print("ERROR: " + str(e))

# COMMAND ----------

# MAGIC %md
# MAGIC Why did this fail?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding Pre-Processing Steps
# MAGIC 
# MAGIC We trained our `rf2` model using a pre-processed training set that has one extra column (`review_scores_sum`) than the unprocessed `X_train` and `X_test` DataFrames.  The `rf2` model is expecting to have `review_scores_sum` as an input column as well. Even if `X_test` had the same number of columns as the processed data we trained on, the line above will still error since it does not have our custom truncated `trunc_lat` and `trunc_long` columns.
# MAGIC 
# MAGIC To fix this, we could manually re-apply the same pre-processing logic to the `X_test` set each time we wish to use our model. 
# MAGIC 
# MAGIC However, there is a cleaner and more streamlined way to account for our pre-processing steps. We can define a custom model class that automatically pre-processes the raw input it receives before passing that input into the trained model's `.predict()` function. This way, in future applications of our model, we will no longer have to worry about remembering to pre-process every batch of data beforehand.
# MAGIC 
# MAGIC Complete the `preprocess_input(self, model_input)` helper function of the custom `RF_with_preprocess` class so that the random forest model is always predicting off of a DataFrame with the correct column names and the appropriate number of columns.

# COMMAND ----------

# TODO
# Define the model class
class RF_with_preprocess(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_rf):
        self.rf = trained_rf

    def preprocess_input(self, model_input):
        '''return pre-processed model_input'''
        
        # Adding the lines from code we have written above
        model_input["trunc_lat"] = model_input["latitude"].round(2)
        model_input["trunc_long"] = model_input["longitude"].round(2)
        
        model_input["review_scores_sum"] = model_input[['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']].sum(axis=1)
        
        model_input = model_input.drop(columns=['latitude','longitude'], axis=1)
        
        return model_input
    
    def predict(self, context, model_input):
        processed_model_input = self.preprocess_input(model_input.copy())
        return self.rf.predict(processed_model_input)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save, then load this custom model's `python_function`.

# COMMAND ----------

# Construct and save the model
model_path =  f"{workingDir}/RF_with_preprocess_/"

dbutils.fs.rm(model_path, True) # remove folder if already exists

rf_preprocess_model = RF_with_preprocess(trained_rf = rf2)
mlflow.pyfunc.save_model(path=model_path.replace("dbfs:", "/dbfs"), python_model=rf_preprocess_model)

# Load the model in `python_function` format
loaded_preprocess_model = mlflow.pyfunc.load_model(model_path.replace("dbfs:", "/dbfs"))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can directly give our loaded model the unmodified `X_test` and have it generate predictions without errors!

# COMMAND ----------

# Apply the model
loaded_preprocess_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding Post-Processing Steps
# MAGIC 
# MAGIC Now suppose we are not as interested in a numerical prediction as we are in a categorical label of `Expensive` and `Not Expensive` where the cut-off is above a price of $100. Instead of retraining an entirely new classification model, we can simply add on a post-processing step to our custom model so it returns the predicted label instead of numerical price.
# MAGIC 
# MAGIC Complete the following model class with **both the previous preprocess steps and the new `postprocess_result(self, result)`** function such that passing in `X_test` into our model will return an `Expensive` or `Not Expensive` label for each row.

# COMMAND ----------

# TODO
# Define the model class
class RF_with_postprocess(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_rf):
        self.rf = trained_rf

    def preprocess_input(self, model_input):
        '''return pre-processed model_input'''
        
        model_input["trunc_lat"] = model_input["latitude"].round(2)
        model_input["trunc_long"] = model_input["longitude"].round(2)
        
        model_input["review_scores_sum"] = model_input[['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']].sum(axis=1)
        
        model_input = model_input.drop(columns=['latitude','longitude'], axis=1)
        
        return model_input
      
    def postprocess_result(self, results):
        '''return post-processed results
        Expensive: predicted price > 100
        Not Expensive: predicted price <= 100'''
        
        out = ['Expensive' if x > 100 else 'Not Expensive' for x in results]
        return out
    
    def predict(self, context, model_input):
        processed_model_input = self.preprocess_input(model_input.copy())
        results = self.rf.predict(processed_model_input)
        return self.postprocess_result(results)

# COMMAND ----------

# MAGIC %md
# MAGIC Create, save, and apply the model to `X_test`.

# COMMAND ----------

# Construct and save the model
model_path =  f"{workingDir}/RF_with_postprocess_/"

dbutils.fs.rm(model_path, True) # remove folder if already exists

rf_postprocess_model = RF_with_postprocess(trained_rf = rf2)
mlflow.pyfunc.save_model(path=model_path.replace("dbfs:", "/dbfs"), python_model=rf_postprocess_model)

# Load the model in `python_function` format
loaded_postprocess_model = mlflow.pyfunc.load_model(model_path.replace("dbfs:", "/dbfs"))

# Apply the model
loaded_postprocess_model.predict(X_test)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Given any unmodified raw data, our model can perform the pre-processing steps, apply the trained model, and follow the post-processing step all in one `.predict` function call!

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Cleanup"

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
