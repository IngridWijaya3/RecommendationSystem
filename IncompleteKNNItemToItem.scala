package cs550.FinalProject

object RecommendationSystem extends App{
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
//import com.johnsnowlabs.nlp.base._
//import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
//import com.johnsnowlabs.nlp.annotators._
//import org.apache.spark.ml.Pipeline
//import com.johnsnowlabs.nlp.RecursivePipeline
//import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.sql.functions
 /*def double getSimilarity(List<? extends Number> thisList, List<? extends Number> thatList) {
        // compute similarity

        if (thisList == null || thatList == null || thisList.size() < 2 || thatList.size() < 2 || thisList.size() != thatList.size()) {
            return Double.NaN;
        }

        double thisMu = Maths.mean(thisList);
        double thatMu = Maths.mean(thatList);

        double num = 0.0, thisPow2 = 0.0, thatPow2 = 0.0;
        for (int i = 0; i < thisList.size(); i++) {
            double thisMinusMu = thisList.get(i).doubleValue() - thisMu;
            double thatMinusMu = thatList.get(i).doubleValue() - thatMu;

            num += thisMinusMu * thatMinusMu;
            thisPow2 += thisMinusMu * thisMinusMu;
            thatPow2 += thatMinusMu * thatMinusMu;
        }

        return num / (Math.sqrt(thisPow2) * Math.sqrt(thatPow2));
    }
*/
//def pearsonSimilarity (original: String, maxLength: Int) : Double = {
  //  return "Not implemented yet";/
//}
//var documentAssembler = new DocumentAssembler()
//.setInputCol("text")
//.setOutputCol("document")

//val sentenceDetector = new SentenceDetector()
//  .setInputCols("document")
 // .setOutputCol("sentence")
  //val parsed = spark.read.option("header", "true").option("nullValue", "?").option("inferSchema", "true").csv("linkage")
   System.setProperty("spark.executor.memory", "40g")
   // System.setProperty("spark.executor.memory", "8g")
     System.setProperty("spark.driver.maxResultSize", "40g")
      System.setProperty("spark.driver.memory", "40g")
       System.setProperty("spark.executor.cores", "15")
     // System.setProperty("spark.memory.useLegacyMode","true")
     // System.setProperty("spark.storage.memoryFraction","0.0")
val conf = new SparkConf()
      .setAppName("RecommendationSystem")
      .setMaster("local[*]")
   // conf.setMaster("yarn")
  //  conf.set(key, value)
   val sc = new SparkContext(conf)
  val confstr=conf.getAll
  confstr.foreach(f=> println(f._1)  )
  confstr.foreach(f=> println(f._2)  )
 //sc.stop()
   //sc.set("spark.executor.memory", "12g")
   // sc.set("spark.executor.cores", "15")
   //sc.set("spark.driver.memory", "8g")     
   // spark.executor.memory  10g
   val sqlContext = new org.apache.spark.sql.SQLContext(sc)
   val spark = org.apache.spark.sql.SparkSession.builder
        .master("local[*]")
        .appName("Spark CSV Reader")
        .getOrCreate;
   import  spark.implicits._
   /*
   var data = spark.read
        .option("header", "false")
        .option("nullValue", "?")
        .option("inferSchema", "true")
        .csv("hdfs://sandbox-hdp.hortonworks.com:8020/user/finalproject/ratings_Clothing_Shoes_and_Jewelry.csv")
   
   
   */
    var data = spark.read
        .option("header", "true")
        .option("nullValue", "?")
        .option("inferSchema", "true")
        .csv("hdfs://sandbox-hdp.hortonworks.com:8020/user/finalproject/ratings_Clothing_Shoes_and_Jewelry_5.csv")


  data.take(10).foreach(println)
  // val productmetadata = spark.read.json("hdfs://sandbox-hdp.hortonworks.com:8020/user/finalproject/meta_Clothing_Shoes_and_Jewelry.json")
  // val reviewdata = spark.read.json("hdfs://sandbox-hdp.hortonworks.com:8020/user/finalproject/reviews_Clothing_Shoes_and_Jewelry_5.json")
   val columnname = Seq("user_id_str", "item_id_str", "rating")//, "time_stamp")
    //reviewdata.printSchema
   //val onlyrating=reviewdata.select("reviewerID","asin", "overall")
   //val newNames = Seq("user_id_str", "item_id_str", "rating")
   //val dfRenamed = onlyrating.toDF(newNames: _*)
   //dfRenamed.write.mode(SaveMode.Overwrite).csv("hdfs://sandbox-hdp.hortonworks.com:8020/user/finalproject/ratings_Clothing_Shoes_and_Jewelry_5.csv")
   data = data.toDF(columnname: _*)
        var userdf = data.select("user_id_str").distinct()
        val inputRows = userdf.rdd.zipWithUniqueId.map{
        case (r: Row, id: Long) => Row.fromSeq(id +: r.toSeq)}
    userdf = sqlContext.createDataFrame(inputRows, StructType(StructField("user_id", LongType, false) +: userdf.schema.fields))
   data= data.join(userdf, data.col("user_id_str") === userdf.col("user_id_str"))
 
   
     var itemdf = data.select("item_id_str").distinct()
        val iteminputRows = itemdf.rdd.zipWithUniqueId.map{
        case (r: Row, id: Long) => Row.fromSeq(id +: r.toSeq)}
    itemdf = sqlContext.createDataFrame(iteminputRows, StructType(StructField("item_id", LongType, false) +: itemdf.schema.fields))
   data= data.join(itemdf, data.col("item_id_str") === itemdf.col("item_id_str"))



  // val selectcolumnname = Seq( "user_id","user_id_str", "item_id_str", "rating")
   //data= data.select(data.columns.filter(colName =>  selectcolumnname.contains(colName)  ) .map(colName => new Column(colName)): _* ) 
   data=data.drop("user_id_str")
   data=data.drop("time_stamp")
   data=data.drop("item_id_str")
  
    val Array(traindata,testdata)=data.randomSplit(Array(0.8,0.1))
    //traindata.cache()
    //testdata.cache()
    
    //println(traindata.getClass())
    
   
    
  // val predictions = model.transform(testdata).withColumn("prediction", col("prediction"))
 
  //predictions.take(50).foreach(println)
   testdata.printSchema
 // reviewdata.take(10).foreach(println)
 // productmetadata.take(10).foreach(println)
 // reviewdata.printSchema
 // productmetadata.printSchema
     testdata.printSchema
    

 var test1data = spark.read
        .option("header", "true")
        .option("nullValue", "?")
       .option("inferSchema", "true")
       .csv("/root/Desktop/ScalaProject/RecommendationSystem/src/data/testdata1.csv")
        test1data = test1data.toDF(Seq("user_id", "item_id", "rating"): _*)
test1data= test1data.withColumn( "user_id", test1data("user_id").cast(LongType) )
  test1data=test1data.withColumn( "item_id", test1data("item_id").cast(LongType) )
 test1data=test1data.withColumn( "rating", test1data("rating").cast(DoubleType) )
         
   //var matrixentry =traindata.map( lambda r:case Row( rating: Double, user_id: Long, item_id: Long)  => MatrixEntry(user_id, item_id , rating))
 //val mat = new CoordinateMatrix(matrixentry.rdd )
   // val mat = new CoordinateMatrix(traindata.map {
  //  case Row(rating: Double, user_id: Long, item_id: Long) => MatrixEntry(user_id, item_id, rating)
//})
   // mat = CoordinateMatrix(traindata.rdd.map(lambda r: MatrixEntry(r.userId, r.itemId, r.rating)))
     //CORRECT 
     sqlContext.dropTempTable("data")
       data.printSchema
     data.unpersist()  
   // val matrixentry= traindata.rdd.map{ r => r match { case Row(rating: Double, user_id: Long, item_id: Long)  => MatrixEntry(user_id, item_id, rating) } }
  //test1data.printSchema
       sqlContext.dropTempTable("traindata")
       traindata.printSchema
     traindata.unpersist() 
   
    testdata.printSchema
        // test1data.take(20).foreach(println)
       val matrixentry= test1data.rdd.map{ r => r match { case Row(user_id: Long, item_id: Long, rating: Double)  => MatrixEntry(user_id-1, item_id-1, rating) } }
    
    val matrixRating = new CoordinateMatrix(matrixentry )
    matrixRating.toIndexedRowMatrix().rows.collect()(3).vector.toArray(2)=0
    matrixRating.toIndexedRowMatrix().rows.collect()(3).vector.toArray(3)=0
    //val itemcolwmatrix=matrixRating.toIndexedRowMatrix()
 
    val asmlvector=(matrixRating.toIndexedRowMatrix().rows.map(v => v.vector))
   //val vectorlist=asmlvector.collect()
 //  val vectorsize=vectorlist(0).size
  
   //  println(matrixRating.numCols())
    //val matrixentry= traindata.map{ r => r match { case Row(rating: Double, user_id: Long, item_id: Long)  => Sparse (user_id, item_id, rating) } }
   //val correlMatrix=matrixRating.toIndexedRowMatrix().columnSimilarities()
   val correlMatrix: Matrix = Statistics.corr( asmlvector , "pearson")
    //matrixRating.toIndexedRowMatrix().rows.foreach(println )
    //val ids=matrixRating.toIndexedRowMatrix().rows.foreach(r=> println( r.index ))
    val itemrating=matrixRating.transpose().toIndexedRowMatrix().rows
    //cols.foreach(println )
   
   // println(correlMatrix)
    var meanarray = Array.ofDim[Double]( ( matrixRating.numCols()).toInt )
    val rddvector = sc.parallelize( asmlvector.collect().toSeq )
    val itemcolmeans=Statistics.colStats(rddvector).mean
    //val numRates:Long = matrixRating.numCols()*matrixRating.numRows();
    val globalSum:Double=matrixRating.entries.filter(f=>f.value>0).map{case MatrixEntry(row, col, value) => (value)}.sum()
    val numberofratings= matrixRating.entries.filter(f=>f.value>0).count()
    val globalMean:Double = globalSum / numberofratings;
    val numOfItems:Long=matrixRating.numCols()
    

   
     itemcolmeans.foreachActive( (index,mean)  =>   meanarray(index.toInt)= (if(mean>0) mean  else globalMean) ) 
   // println(colkeymean)
    val itemMeans = Vectors.dense(meanarray).asInstanceOf[DenseVector]
//predictions.printSchema
    
  //val evaluator = new RegressionEvaluator()
 // .setMetricName("rmse")
 // .setLabelCol("rating")
 // .setPredictionCol("prediction")
//val rmse = evaluator.evaluate(predictions)
//println(s"Root-mean-square error = $rmse")
     println("==============")
      //println(traindata.getClass())
      // println("to row matrix")
   // println("cols : ",  matrixRating.numCols())
   // println(vectorlist.getClass())
 
 


 
    var  itemSimilarityListtem = Array.ofDim[(Long,Array[Double])](  correlMatrix.numRows.toInt )
    val numItems= ( matrixRating.numCols()).toInt
    var itemIndex:Int=0
   
    correlMatrix.rowIter.foreach( similarityRow =>
      {
           
            itemSimilarityListtem(itemIndex)=(itemIndex ,Array.ofDim[Double](similarityRow.size ))
            val sortedsData=similarityRow.argmax
            
            similarityRow.foreachActive((simindex,simval) => {
               //println(simval)
                try
                 {
               if(itemSimilarityListtem(itemIndex)!=null && itemSimilarityListtem(itemIndex)!=null && similarityRow.size>0)// && itemSimilarityListtem(itemIndex)._2.size>0)
               {
           
                 //println( itemSimilarityListtem(itemIndex)._2)
                 itemSimilarityListtem(itemIndex)._2(simindex)=simval
                
                 
               }
                }
                 catch
                 {
                   case ex: NullPointerException => {
                        println("There are no elements in this list.")
                          println( itemSimilarityListtem.size)
                           println( itemIndex)
                            println( simval)
                             println( simindex)
                              println(  itemSimilarityListtem(itemIndex))
                         // println(itemSimilarityListtem(itemIndex)._2)    
                   }
                 }
            })
           itemIndex=itemIndex+1
  
           // Lists.sortList(userSimilarityList[userIndex], true);
        })
    
    val  itemSimilarityList =  sc.parallelize( Seq(itemSimilarityListtem))
    //itemSimilarityList.cache()
    println("====================")
    println("item similariryt " )
    itemSimilarityList.collect().foreach(f => f.toArray.foreach(println))
    println("====================")
    var nns= scala.collection.mutable.Map[Int, Double]()
           val userIdx=2
           val itemindex:Int=3
      //var nns = new ArrayList<>();
      val users=matrixRating.toIndexedRowMatrix().rows
       itemSimilarityList.collect().foreach(f => f.toArray.foreach(t=> println(t._1)))
      var simList = itemSimilarityList.first().filter(f => f._1== itemindex).toArray
       println(simList.size )
      println("similarity list for 3")
    simList.foreach(f =>  f._2.foreach( println) ) 
 
     println("==============")
    println(" global mean : ", globalMean )
     println(" global sum : ", globalSum )
      println(" number of ratings : ", numberofratings )
     itemcolmeans.foreachActive((ind,mean) => println(ind+" , "+mean  )   )
   //itemcolwmatrix.rows.foreach(f=> println(f.index))
   println("==============")
   println(correlMatrix)
    println("====================")
        var count:Int = 0;
        
        val usercurrentrow = users.filter(f=> f.index== userIdx).first()
        var currentItemIdxSet = scala.collection.mutable.ArrayBuffer.empty[Int]
        usercurrentrow.vector.foreachActive((index,ratingvalue)=>  {
          if( ratingvalue >0 )
          {
          currentItemIdxSet+=index
      
          }
               println("==============")
           println(" current item index ")
           println(index)
           println(ratingvalue)
           println(currentItemIdxSet)
        } )
            
        var similarItemIdx:Int=0
        simList.foreach(iter => iter._2.foreach(sim  => { 
           if (currentItemIdxSet.contains(similarItemIdx)) {
             
          if(sim>0)
          {
              nns+= (similarItemIdx -> sim)
                println("==============")
              //println(" nns ")
              // println(nns)
                 println("==============")
          }
           }
        
         similarItemIdx+=1
        }) )

        
        var sum:Double= 0
        var ws:Double = 0;
        nns.foreach(itemRatingEntry => {    
          var similarItemIdx:Int = itemRatingEntry._1
           val sim:Double = itemRatingEntry._2
          // val rate:Double=matrixRating
           
                val rating= matrixRating.toIndexedRowMatrix().rows.filter(f => f.index==userIdx ).first().vector(similarItemIdx) //.vector[1]
                
                sum = sum + ( sim * (rating - itemMeans(similarItemIdx)));
                ws += Math.abs(sim);
                  println("==============")
                  println("sum")
                println(sum)
                println("similarItemIdx")
                println(similarItemIdx)
                 println("userIdx")
                println(userIdx)
                println("rating")
                println(rating)
                 println("itemindex")
                println(itemindex)
                 println("itemMeans(similarItemIdx)")
                println(itemMeans(similarItemIdx))
                println("ws")
                println(ws)
                var ratingprediction =if (ws > 0)  itemMeans(itemindex) + sum / ws else  globalMean
                 println(ratingprediction)
                
                println("==============")
                
        } )
            
       /*

        if (validMatrix == null)
            return false;

        // get posterior probability distribution first
        estimateParams();

        // compute current RMSE
        int numCount = 0;
        double sum = 0;
        for (MatrixEntry me : validMatrix) {
            double rate = me.get();

            int u = me.row();
            int j = me.column();

            double pred = 0;
            try {
                pred = predict(u, j, true);
            } catch (LibrecException e) {
                e.printStackTrace();
            }
            if (Double.isNaN(pred))
                continue;

            double err = rate - pred;

            sum += err * err;
            numCount++;
        }

        double RMSE = Math.sqrt(sum / numCount);
        double delta = RMSE - preRMSE;

        if (numStats > 1 && delta > 0)
            return true;

        preRMSE = RMSE;
        return false;*/
     println("================")
     //colkeymean.foreach(f=> println(f._1 +" , "+f._2))
     sc.stop()
   
}