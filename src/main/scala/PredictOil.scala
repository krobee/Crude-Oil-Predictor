package main.scala

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.FloatType
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler

object PredictOil{
 
  def main(args: Array[String]) {
    
    // local mode for testing
    val spark = SparkSession.builder
      .master("local")
      .appName("PredictOil")
      .getOrCreate
    
    // cluster mode for deploying
//    val sc = SparkSession.builder
//             .master("spark://jupiter:31106")
//             .appName("PredictOil")
//             .getOrCreate
             
    // read WTI oil price
    val dfOilRaw = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load("hdfs://jupiter:31101/cs555_data/oil_price.csv"); //.csv("csv/file/path")
    
    // make date format consistent with other datasets
    var dfOil = dfOilRaw.withColumn("DATE", regexp_replace(col("DATE"), "-", ""))
    
    // convert column type
    dfOil = dfOil.withColumn("DATE", col("DATE").cast(IntegerType))
      .withColumn("PRICE", col("PRICE").cast(FloatType))
    
    // read GDELT events
    var dfEvents = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load("hdfs://jupiter:31101/cs555_data/events_1101_1120.csv"); //.csv("csv/file/path")
    
    // convert column type
    dfEvents = dfEvents.withColumn("GLOBALEVENTID", col("GLOBALEVENTID").cast(IntegerType))
      .withColumn("SQLDATE", col("SQLDATE").cast(IntegerType))
      .withColumn("GoldsteinScale", col("GoldsteinScale").cast(FloatType))
      .withColumn("AvgTone", col("AvgTone").cast(FloatType))
      .withColumn("NumSources", col("NumSources").cast(FloatType))
    
    // read GDELT total sources
    var dfTotalSources = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load("hdfs://jupiter:31101/cs555_data/total_sources_1101_1120.csv"); //.csv("csv/file/path")
    
    // convert column type
    dfTotalSources = dfTotalSources.withColumn("SQLDATE", col("SQLDATE").cast(IntegerType))
      .withColumn("TotalSources", col("TotalSources").cast(FloatType))
    
    // merge gdelt datasets
    var dfEventsWithTotal = dfEvents.join(dfTotalSources, "SQLDATE")
   
    // data pre-processing
    
    // data structure <SQLDATE  |  sum(GoldsteinScale * (NumSources / TotalSources))  |  sum(AvgTone * (NumSources / TotalSources))>
    dfEventsWithTotal = dfEventsWithTotal.groupBy("SQLDATE")
      .agg(sum(col("GoldsteinScale") * (col("NumSources") / col("TotalSources"))).alias("SumGold"),
           sum(col("AvgTone") * (col("NumSources") / col("TotalSources"))).alias("SumAvgTone"))
      .orderBy(asc("SQLDATE"))
    
//  dfEventsWithTotal.show()
    
    // merge with oil price
    var dfEventsWithPrice = dfOil.join(dfEventsWithTotal, col("SQLDATE") === col("DATE"))
      .drop(col("DATE"))
      .drop(col("SQLDATE"))
//    dfEventsWithPrice.show()
  
    // convert to dense vector feature
    val assembler = new VectorAssembler()
      .setInputCols(Array("SumGold", "SumAvgTone"))
      .setOutputCol("features")
      
    val data = assembler.transform(dfEventsWithPrice)
      .drop(col("SumGold"))
      .drop(col("SumAvgTone"))
      .withColumnRenamed("PRICE", "label")
      
//    data.show()
    
    
    // split training and test dataset
    val splits = data.randomSplit(Array(0.8,0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()
    
    val numTraining = training.count()
    val numTest = test.count()
    
    // training
    val lr = new LinearRegression()
    val lrModel = lr.fit(training)
    
    // testing
    val predicted = lrModel.transform(test)
    predicted.show()
    
//    println(s"Training: $numTraining, test: $numTest.")
//    
//    // Print the coefficients and intercept for linear regression
//    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
//
//    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
    
    
    
    
    
    
    
    spark.stop()
  }
}