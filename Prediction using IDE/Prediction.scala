import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark._

object Prediction extends App {
  val conf = new SparkConf().setMaster("local[*]").setAppName("HeartFailurePrediction")
  val sc = new SparkContext(conf)
  val spark = SparkSession.builder().getOrCreate()
  val data = spark.read.format("csv").option("header", true).option("inferschema", true).load("hdfs://localhost:9000/Datasets/heart_failure.csv")
  val labeledDf = data.withColumnRenamed("DEATH_EVENT", "label")
  val featureColumns = Array("age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time")
  val assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features")
  val Array(training, test) = labeledDf.randomSplit(Array(0.75, 0.25), seed = 4321)
  val logisticRegression = new LogisticRegression()
  val stages = Array(assembler, logisticRegression)
  val pipeline = new Pipeline().setStages(stages)
  val model = pipeline.fit(training)
  val predictedResults = logisticModel.transform(test)
  val predictionAndLabel = predictedResults.select($"prediction", $"label").as[(Double, Double)].rdd 
  val metrics = new MulticlassMetrics(predictionAndLabel)
  println(metrics.confusionMatrix)
  metrics.accuracy
}
