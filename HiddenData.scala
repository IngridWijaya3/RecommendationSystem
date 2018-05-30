package cs550.FinalProject

object HiddenData extends App{
 
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
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.SaveMode;
  System.setProperty("spark.executor.memory", "20g")
  System.setProperty("spark.driver.maxResultSize", "20g")
  System.setProperty("spark.driver.memory", "20g")
  System.setProperty("spark.executor.cores", "15")
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
   
    val parse_column: (Column) => Column = (x) => { split(x, ",")(1) }
   
    var imlitcidata = spark.read.option("header", true).csv("/../gradpool/Desktop/musicInst_metadata2.csv")
    //df = df.withColumn("city", parse_city(col("location")))
   //imlitcidata.map(func => func)
  /* val alsoView= imlitcidata.select("also_viewed").rdd.foreach(f => {
      var alsoviewed= f.getAs[String]("also_viewed") //("also_viewed")
      if(alsoviewed!=null)
      {
         alsoviewed.replaceAll("""\[\]""","").split(",").foreach( println )
      }
   })
   */
    val cols= Seq( "item_id","price","brand", "user_also_view")
   val alsoView= imlitcidata.filter( ($"also_viewed".isNotNull)).flatMap(f => {
      var alsoviewed= f.getAs[String]("also_viewed") //("also_viewed")
     // var title= f.getAs[String]("title") //("also_viewed")
      var price = f.getAs[String]("price")
      val asin= f.getAs[String]("asin") //("also_viewed")
      val brand= f.getAs[String]("brand")
      val salesRank=f.getAs[String]("salesRank")
   
         alsoviewed.replaceAll("[\\[\\]]","").split(",").map(s=> ( asin ,price,brand,  s))
      

   }).toDF(cols: _*)
   val buy_after_viewing =imlitcidata.filter( ($"buy_after_viewing".isNotNull)).flatMap(f => {
      var buy_after_viewing= f.getAs[String]("buy_after_viewing") //("also_viewed")
    
      val asin= f.getAs[String]("asin") //("also_viewed")
     
   
         buy_after_viewing.replaceAll("[\\[\\]]","").split(",").map(s=> ( asin,   s))
      

   }).toDF(Seq( "item_id", "user_buy_after_viewing"): _*)

   var combine=alsoView.join( buy_after_viewing, alsoView.col("item_id") === buy_after_viewing.col("item_id") && alsoView.col("user_also_view") === buy_after_viewing.col("user_buy_after_viewing") , "left")
   .toDF( Seq( "item_id", "price","brand", "user_also_view","item_id_2","user_buy_after_viewing") : _*)
    combine= combine.drop( "item_id_2")
  // alsoView.printSchema()
  //  combine.printSchema()
    
    /*combine.foreach(f=> {
      
      var buy_after_viewing= f.getAs[String]("user_buy_after_viewing") //("also_viewed")
      var alsoviewed= f.getAs[String]("user_also_view")
      val asin= f.getAs[String]("item_id") //("also_viewed")
      println("=================")
      println(asin)
       println(buy_after_viewing)
        println(alsoviewed)
          println("=================")
      
    })
    */
    val buyafterviewinnusers =combine.filter ($"user_buy_after_viewing".isNotNull)
     //buyafterviewinnusers.write.format("csv").save("/../gradpool/Desktop/user_buy_after_view.csv")
   buyafterviewinnusers.repartition(1)
   .write.format("com.databricks.spark.csv")
   .option("header", "true")
   .save("/../gradpool/Desktop/user_buy_after_view.csv")
    val notbuyafterviewinnusers2 =combine.filter($"user_buy_after_viewing".isNull)
    notbuyafterviewinnusers2.repartition(1)
   .write.format("com.databricks.spark.csv")
   .option("header", "true")
   .save("/../gradpool/Desktop/user_notbuy_after_view.csv")
    //notbuyafterviewinnusers2.write.format("csv").save("/../gradpool/Desktop/user_notbuy_after_view.csv")
     println("=================")
    println(combine.count())
     println("=================")
     println("=================")
    //println(test.count(),combine.count() )
     println("=================")
     ////
   /* test.rdd.foreach(f=> {
       var buy_after_viewing= f.getAs[String]("user_buy_after_viewing") //("also_viewed")
      var alsoviewed= f.getAs[String]("user_also_view")
      val asin= f.getAs[String]("item_id") //("also_viewed")
    })*/
    sc.stop()
}