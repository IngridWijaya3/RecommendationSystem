from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import struct
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.ml.linalg import Vectors, VectorUDT


from sklearn.metrics.pairwise import cosine_similarity


def read_to_rdd(path):
    filenameRdd = sc.binaryFiles(path)
    print(filenameRdd.collect())
    return filenameRdd

def map_rdd_file(filenameRdd):
    key = ''
    value = []
    for i in xrange(1503384):
        if i % 4097 == 0:
            key = filenameRdd.collect()[i]
            value.append(filenameRdd.collect()[i+1:i+1+4096])
            print((key,value))
    return (key,value)


def readImageFeatures(path):
    f = open(path, 'rb')
    k = 0
    a_f_l = []

    while True:
        #print("we are reading the-------------------------------- " + str(k) +" -----image")
        k+=1
        asin = f.read(10)
        if asin == '': break
        feature = []
        for i in range(4096):
            #print(struct.unpack('f', f.read(4))[0])
            feature.append(struct.unpack('f', f.read(4)))
            #print("read " + str(i) + "features" + "of image : " + asin )

        a_f_l.append((asin,feature,k))
        print(k)
        print(asin)
    #print(a_f_l[68][0])
    print(len(a_f_l))
    #a_f_l = sc.parallelize(a_f_l, numSlices=5000)
    #a_f_l = sc.parallelize(a_f_l).sample(False, 0.5, 1234)

    #train = sc.parallelize(a_f_l[0:63]).cache()
    #test  = sc.parallelize(a_f_l[64:100]).cache()

    #a_f_l = sc.parallelize(a_f_l).cache()
    #return a_f_l
    #return train, test

#no need Load training data
#data = spark.read.format("libsvm")\
#    .load("data/mllib/sample_multiclass_classification_data.txt")

def split_train_test(data):
    # Split the data into train and test
    splits = data.randomSplit([0.6, 0.4])
    train = splits[0]
    test = splits[1]
    # return train,test

    #train = data.slice(0,65)
    #train,test = data.split([0.6,0.4])
    #print(train)
    #train = splits[0]
    #test = data.split([65,99])
    #print(test)
    return train,test

def mpc(train, test):
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = [4096, 1028, 256, 100]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)

    # train the model
    model = trainer.fit(train)

    # compute accuracy on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    print(result)
    result.show()

    print('result')
    print(predictionAndLabels)
    #return result.toPandas()
    result.toPandas().to_csv('hundred_all.csv')
    result.select('probability').toPandas().to_csv('output_index_36.csv')
    #result.toPandas().to_csv('hundred_all.csv')
    #result.select('probability').toPandas().to_csv('output_all.csv')





if __name__ == "__main__":
    # SparkContext.setSystemProperty('spark.executor.memory', '6g')
    conf = (SparkConf().set("spark.driver.maxResultSize", "7g").set("spark.driver.memory", "7g")
            .set("spark.executor.cores ", "4").set("spark.executor.memory","7g")
            .set("spark.memory.useLegacyMode", "true").set("spark.shuffle.memory","0.1")
            .set("spark.storage.memoryFraction","0.5"))
    # conf = (SparkConf().set("spark.driver.memory", "7g"))
    # conf = (SparkConf().set("spark.executor.cores ","4"))
    # conf = (SparkConf().set("spark.executor.memory","7g"))
    # conf = (SparkConf().set("spark.memory.useLegacyMode","true"))
    # conf = (SparkConf().set("spark.shuffle.memory","0.01"))
    # conf = (SparkConf().set("spark.storage.memoryFraction","0.0"))

    sc = SparkContext(conf=conf)
    #sc.stop()
    #sc.setSystemProperty('spark.executor.memory','6g')

    sqlContext = SQLContext(sc)
    path = '/Users/liyaguan/Downloads/image_features_Musical_Instruments.b'

    #predata = readImageFeatures(path)
    #train, test = readImageFeatures(path)

    readImageFeatures(path)

    #predata = predata.map(lambda x: (x[2], x[1]))

    train = train.map(lambda x: (x[2], x[1]))
    test = test.map(lambda x: (x[2], x[1]))

    cschema = StructType([
        StructField("asin", StringType(), True),
        StructField("feature", FloatType(), True)
    ])

    #index = sc.parallelize(xrange(100))

    #DF = sqlContext.createDataFrame(predata)
    #df = predata.map(lambda x: (x[0],Vectors.dense(x[1])))
    train = train.map(lambda x: (x[0],Vectors.dense(x[1])))
    test = test.map(lambda x: (x[0],Vectors.dense(x[1])))

    #df = df.toDF(['label','features'])

    train = train.toDF(['label','features'])
    test = test.toDF(['label', 'features'])
    #DF = DF.rdd.map(lambda x: x.toDF(['label','features']))
    #df.show()
    #DF.show()


    #print(predata.take(1))

    #splitted = split_train_test(df)
    #mpc(splitted[0],splitted[1])

#    mpc(train, test)




    #cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    #array([[1., 0.36651513, 0.52305744, 0.13448867]])

    #filenameRdd = read_to_rdd(path)
    #map_rdd_file(filenameRdd)
#    print(sc._conf.getAll())
    SparkContext.stop(sc)

