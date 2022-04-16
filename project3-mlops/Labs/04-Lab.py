# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Lab: Runnable Notebooks
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC - Create a parameterized, runnable notebook that logs an MLflow run 
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> This lab makes use of the **`notebooks`** API which is currently unavailable on the Community Edition of Databricks.

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
# MAGIC ## Create a Runnable Notebook
# MAGIC 
# MAGIC [Fill out the notebook in this directory 04-Lab-Runnable.]($./04-Lab-Runnable)  It should accomplish the following:<br><br>
# MAGIC 
# MAGIC 0. Read in 3 parameters:
# MAGIC   - `n_estimators` with a default value of 100
# MAGIC   - `learning_rate` with a default value of .1
# MAGIC   - `max_depth` with a default value of 1
# MAGIC 0. Train and log a gradient boosted tree model
# MAGIC 0. Report back the path the model was saved to

# COMMAND ----------

# MAGIC %md
# MAGIC Run the notebook with the following code.

# COMMAND ----------

notebookrun = dbutils.notebook.run("./04-Lab-Runnable", 60, 
          {"n_estimators": "100",
           "learning_rate": ".1",
           "max_depth": "1"})
print(notebookrun)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Check that these logged runs also show up properly on the MLflow UI. 
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See the solutions folder for an example solution to this lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Model Management]($../05-Model-Management) 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
