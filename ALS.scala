package cs550.FinalProject

object ALS extends App{
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.ml.recommendation._
import scala.util.Random
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql._
import org.apache.spark.sql.types.{StructType, StructField, LongType}
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable.ArrayBuffer
import scala.collection.Map
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.rdd
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.sql.functions
import java.io._
import scala.io
import java.io.PrintWriter

  System.setProperty("spark.executor.memory", "10g")
  System.setProperty("spark.driver.maxResultSize", "10g")
  System.setProperty("spark.driver.memory", "10g")
  System.setProperty("spark.executor.cores", "10")
   
     val conf = new SparkConf()
      .setAppName("RecommendationSystem")
      .setMaster("local[*]")
   val sc = new SparkContext(conf)
   val sqlContext = new org.apache.spark.sql.SQLContext(sc)
   val spark = org.apache.spark.sql.SparkSession.builder
        .master("local[*]")
        .appName("RecommendationSystem")
        .getOrCreate;
   import  spark.implicits._
   
  

   var reviewdata = spark.read.json("hdfs://sandbox-hdp.hortonworks.com:8020/user/finalproject/reviews_Musical_Instruments_5.json")
   reviewdata=reviewdata.select(  "reviewerID","asin" ,"overall")
   val columnname = Seq( "user_id_str", "item_id_str","rating")
   reviewdata = reviewdata.toDF(columnname: _*)
   reviewdata.printSchema
   reviewdata.take(10).foreach(println)
   var userdf =  reviewdata.select("user_id_str").distinct()
  //generate integer 
   val inputRows = userdf.rdd.zipWithUniqueId.map{ case (r: Row, id: Long) => Row.fromSeq(id +: r.toSeq)}
   userdf = sqlContext.createDataFrame(inputRows, StructType(StructField("user_id", LongType, false) +: userdf.schema.fields))
   var itemdf = reviewdata.select("item_id_str").distinct() 
   val iteminputRows = itemdf.rdd.zipWithUniqueId.map{case (r: Row, id: Long) => Row.fromSeq(id +: r.toSeq)}
   itemdf = sqlContext.createDataFrame(iteminputRows, StructType(StructField("item_id", LongType, false) +: itemdf.schema.fields))
   reviewdata= reviewdata.join(itemdf, reviewdata.col("item_id_str") === itemdf.col("item_id_str"))
   reviewdata= reviewdata.join(userdf, reviewdata.col("user_id_str") === userdf.col("user_id_str"))
   
   reviewdata=reviewdata.drop("user_id_str")
   reviewdata=reviewdata.drop("time_stamp")
   reviewdata=reviewdata.drop("item_id_str")
  
 
  val Array(traindata,testdata)=reviewdata.randomSplit(Array(0.8,0.1))
  traindata.cache()
  testdata.cache()

    
    val als = new ALS()
    .setSeed(Random.nextLong())
    .setImplicitPrefs(true)
    .setRank(10)
    .setRegParam(0.01)
    .setAlpha(1.0)
    .setMaxIter(10)
    .setUserCol("user_id")
    .setItemCol("item_id")
    .setRatingCol("rating")
    .setPredictionCol("prediction")
    

    var err = 0
    var min_error = Double.PositiveInfinity 
    var best_rank = -1
    //val model=als.fit(traindata)
    var j:Int=0
    var i:Int=0
    var bestRankIndex:Int=0
    var bestRegIndex:Int=0
    
   
   //Seq(1.0, 0.0001)
    var results=for (rank<-  2 to 15 ;
           reg <-  Array(1,0.025,0.001) ;
           alpha    <-  1 to 10)
      yield {
              als.setAlpha(alpha)
              als.setRank(rank)
              als.setRegParam(reg)
              var model = als.fit(traindata)
              val predictions = model.transform(testdata).withColumn("prediction", col("prediction"))
              val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
              val rmse = evaluator.evaluate(predictions)
   
              predictions.printSchema
         
              min_error= if(rmse < min_error) rmse else min_error  
              bestRankIndex = if(rmse < min_error) i else bestRankIndex  
              bestRegIndex= if(rmse < min_error) j else bestRegIndex  
             
                      val fw2 = new FileWriter("/../gradpool/Desktop/alsresult_reg.txt", true)
try {
  fw2.write( rank+" ,"+reg + "," + alpha +"," + rmse)
  fw2.write("\n")

}
finally fw2.close()
            println( rank ,reg,alpha, rmse)
              (rank, reg,alpha, rmse ,model)
          

          }
         
        

    println(s"Smalest Error = " + min_error )
    results=results.sortBy(f => f._3)//.foreach(println)
    
 val fw3 = new FileWriter("/../gradpool/Desktop/sortedalsresult2.txt", true)
try {
  results.foreach(f=> {
      fw3.write( f._1 +" ,"+ f._2 + "," +  f._3 +"," +  f._4)
  fw3.write("\n")
  })


}
finally fw3.close()

       
    val bestrank=results(0)._1
    val bestreg=results(0)._2
     val bestalpha =results(0)._3
     als.setAlpha(bestalpha)
     als.setRegParam(bestreg)
     als.setRank(bestrank)
    var model = als.fit(traindata)
    val predictions = model.transform(testdata).withColumn("prediction", col("prediction")).withColumnRenamed("prediction", "TruePositive")

      
    val falsePositiveeData = testdata.select("user_id", "item_id").as[(Long,Long)].groupByKey { case (user, _) => user }.
      flatMapGroups { case (userID, userIDitemIDs) =>
        val random = new Random()
        val trueItemIdSet = userIDitemIDs.map { case (_, item) => item }.toSet
        val falsePositive = new ArrayBuffer[Long]()
        val listofItemIDs = reviewdata.select("item_id").as[Long].distinct().collect()
        var i = 0
    
        while (i < listofItemIDs.length && falsePositive.size < trueItemIdSet.size) {
          val itemID = listofItemIDs(random.nextInt(listofItemIDs.length))
        
          if (!trueItemIdSet.contains(itemID)) {
            falsePositive += itemID
          }
          i += 1
        }
        // Return the set with user ID added back
        falsePositive.map(itemID => (userID, itemID))
      }.toDF("user_id", "item_id")

    val falseoPredictions =  model.transform(falsePositiveeData).withColumnRenamed("prediction", "falsePrediction")

    val joinedPredictions = predictions.join(falseoPredictions, "user_id").select("user_id", "TruePositive", "falsePrediction").cache()

    
   val allCounts = joinedPredictions.groupBy("user").agg(count(lit("1")).as("total")).select("user_id", "total")

    val correctCounts = joinedPredictions.filter($"TruePositive" > $"falsePrediction").groupBy("user").agg(count("user_id").as("correct")).select("user", "correct")

    
    val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")). agg(mean("auc")).as[Double].first()

    joinedPredictions.unpersist()

    println(meanAUC)
    
    
     val fw4 = new FileWriter("/../gradpool/Desktop/AUCresult2.txt", true)
try {
 
  fw4.write( meanAUC.toString())
  fw4.write("\n")
 

}
finally fw4.close()

  //reviewdata.printSchema
  reviewdata.unpersist()  

    traindata.unpersist()
  
    sc.stop()
  
}