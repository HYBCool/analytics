package com.hyb

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import scala.util.Random

object DecisionTree {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("decision tree")
      .setMaster("local[4]")
      .set("spark.debug.maxToStringFields", "1000")
    val spark = SparkSession.builder.config(conf).getOrCreate()

    val dataWithoutHeader = spark.read
      .option("inferSchema", "true")
      .option("header", "false")
      .csv("G:\\data\\covtype.data")

    val colNames = Seq(
      "Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_RoadWays",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    ) ++ (
      (0 until 4).map(i => s"Wilderness_Area_$i")
      ) ++ (
      (0 until 40).map(i => s"Soil_Type_$i")
      ) ++ Seq("Cover_Type")

    var data = dataWithoutHeader.toDF(colNames: _*).withColumn("Cover_Type", col("Cover_Type").cast("double"))

    val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    testData.cache()

    val inputCols = trainData.columns.filter(_ != "Cover_Type")
    var assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")

    var assemblerTrainData = assembler.transform(trainData)
    //assemblerTrainData.select("featureVector").show()

    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")

    val model = classifier.fit(assemblerTrainData)
    //println(model.toDebugString)
    //model.featureImportances.toArray.zip(inputCols).sorted.reverse.foreach(println)

    val predictions = model.transform(assemblerTrainData)
    predictions.select("Cover_Type", "prediction", "probability").show(truncate = false)

    val evaluator=new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
    val accuracy=  evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1=evaluator.setMetricName("f1").evaluate(predictions)
    println(accuracy)
    println(f1)

    val confusionMatrix =predictions
      .groupBy("Cover_Type")
      .pivot("prediction",(1 to 7))
      .count()
      .na.fill(0)
      .orderBy("Cover_Type")
    confusionMatrix.show()

    val trainPriorProbabilities=classProbabilities(trainData)
    val testPriorProbabilities=classProbabilities(testData)
    val randomAccuracy=trainPriorProbabilities.zip(testPriorProbabilities).map{
      case (trainProb,cvProb)=>trainProb*cvProb
    }.sum
    println(randomAccuracy)

  }

  def classProbabilities(data:DataFrame):Array[Double]={
    import data.sparkSession.implicits._

    val total=data.count()
    data.groupBy("Cover_Type").count()
      .orderBy("Cover_Type")
      .select("count").as[Double]
      .map(_/total)
      .collect()
  }


}
