
!pip install pyspark

#import necessary libraries
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, expr, when

from pyspark.ml.feature import  VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#create a spark session
spark = SparkSession.builder.appName('Churn_Model').getOrCreate()
#read the dataset
df = spark.read.csv('telecom_dataset.csv', header=True, inferSchema=True)

#inspect the df
df.show()

#view the schema
df.printSchema()

"""## Data Pre-processing

Perform necessary preprocessing steps on the dataset, including
handling missing values.
"""

#drop nulls
df = df.na.drop()
#drop duplicates
df = df.dropDuplicates()

#Convert the numerical columns  from string to int
numerical = ['CustomerID', 'Age', 'MonthlyCharges', 'TotalCharges' ]
for column in numerical:
  df = df.withColumn(column, col(column).cast('double'))     

#check schema to confirm conversion
df.printSchema()

"""## Feature Engineering
Create new features from the existing dataset that might be helpful for predicting churn. We will create age group and  customer tenure columns

"""

#create a new column Age_group 
age_grouped_df = df.withColumn("Age_group",
                   when(col("Age") < 18, "< 18")
                   .when((col("Age") >= 18) & (col("Age") <= 24), "18-24")
                   .when((col("Age") >= 25) & (col("Age") <= 35), "25-35")
                   .when((col("Age") >= 36) & (col("Age") <= 55), "35-55")
                   .otherwise("> 55"))

#create a new column customer tenure by diviving total amount with
tenure_df = age_grouped_df.withColumn("Tenure(Months)", expr('TotalCharges/MonthlyCharges'))
#check schema to confirm dtypes of the newly created columns
tenure_df.printSchema()

# Feature Scaling 
# Define the numerical columns
numerical_columns = ['Age', 'MonthlyCharges', 'TotalCharges']

# Assemble the numerical columns into a vector column
assembler = VectorAssembler(inputCols=numerical_columns, outputCol="vFeatures")

# Create a MinMaxScaler object
scaler = MinMaxScaler(inputCol="vFeatures", outputCol="scaled_features")

# encoding categorical cols
categorical_cols = ['Gender', 'Contract', 'Churn', 'Age_group']
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

pipeline = Pipeline(stages=indexers)
transformed_df = pipeline.fit(tenure_df).transform(tenure_df)
transformed_df.show()

# Splitting the data into training and testing sets
train_data, test_data = transformed_df.randomSplit([0.7, 0.3], seed=42)

#feature columns
feature_cols = ['Gender_index', 'Age', 'Contract_index', 'MonthlyCharges', 'TotalCharges']

# Assemble the numerical columns into a vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_df = assembler.transform(transformed_df)

"""## Model Selection and Training:"""

# Define the models to try
models = [
    RandomForestClassifier(labelCol="Churn_index", featuresCol="features", seed=42),
    LogisticRegression(labelCol="Churn_index", featuresCol="features")
]

# Create a list of parameter grids to search through
paramGrids = [
    ParamGridBuilder().addGrid(RandomForestClassifier.maxDepth, [5, 10]).build(),
    ParamGridBuilder().addGrid(LogisticRegression.regParam, [0.01, 0.1]).build()
]

# Create a list to store the accuracy for each model
accuracies = []

# Train and evaluate each model
for i, model in enumerate(models):
    pipeline = Pipeline(stages=[assembler, model])

    # Set up the cross-validator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrids[i],
                              evaluator=BinaryClassificationEvaluator(labelCol="Churn_index"),
                              numFolds=5)

    # Fit the model and select the best set of parameters
    cvModel = crossval.fit(train_data)
    bestModel = cvModel.bestModel

    # Make predictions on the test data
    predictions = bestModel.transform(test_data)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol="Churn_index")
    accuracy = evaluator.evaluate(predictions)
    accuracies.append(accuracy)

    # Print the accuracy for each model
    print(f"Accuracy for Model {i + 1}: {accuracy}")

# Select the best model based on accuracy
best_model_index = accuracies.index(max(accuracies))
best_model = models[best_model_index]

# Train the best model on the full training data
pipeline = Pipeline(stages=[assembler, best_model])
model = pipeline.fit(assembled_df)

# Save the best model
model.save("Churn Model")

# Closing the SparkSession
spark.stop()