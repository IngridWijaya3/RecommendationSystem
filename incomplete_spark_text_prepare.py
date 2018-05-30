from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
import re
from pyspark.mllib.feature import Word2Vec
from gensim.models import Doc2Vec
from collections import namedtuple
#from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import numpy as np
import statsmodels.api as sm
from random import sample
from timeit import default_timer
import datetime
from collections import OrderedDict
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from random import shuffle

def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

# get rdd data for user review
def get_rdd_data(sc, path):
    data = sc.textFile(path)
    data = data.map(lambda l: l.split('\n')) # split by line
    return data

# filter the stopwords for review and stem it
# pair the processed text with userID
def process_text(eachtext):
    word_tokens = word_tokenize(eachtext[0])
    filtered_sentence = []
    for word in word_tokens:
        if word not in stop_words:
            re.sub('[^A-Za-z0-9]+', '', word)
            if len(word)>1:
                word = stemmer.stem(word.lower())
                filtered_sentence.append(word)
    return (filtered_sentence, eachtext[1])

def process_lable(eachtext):

    return eachtext[0]

def make_tag_doc(text, lable):
    sentences =[]
    alldocs = []
    text = text.collect()
    tags = lable.collect()

    for i in range(len(text)):
        #sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][n//12500]
        #split = ['train', 'test', 'extra', 'extra'][i // 25000]
        #alldocs.append(SentimentDocument(text[i],tags[i],split,sentiment))
        sentences.append(TaggedDocument(words=text[i], tags=tags[i]))
    #train_docs = [doc for doc in alldocs if doc.split == 'train']
    #test_docs = [doc for doc in alldocs if doc.split == 'test']
    #doc_list = alldocs[:]  # For reshuffling per pass
    return sentences #, train_docs, test_docs


# word2vec
# PV-DBOW
# This step takes about 5 minutes on 8GB ram
def wordtovec(wordrdd):
    word2vec = Word2Vec()
    model = word2vec.fit(wordrdd)
    print(model.getVectors())
    synonyms = model.findSynonyms('1',5)
    for word, cosine_distance in synonyms:
        print ("{}: {}".format(word, cosine_distance))

# Evaluating the performance of Doc2Vec
# Classify document sentiments by logistic regression
def logistic_predict(train_target, train_regressor):
    logit = sm.Logit(train_target, train_regressor)
    predictor = logit.fit(disp=0)
    print(predictor.summary())
    return predictor


def error_rate_for_model(test_model,
                         train_set, test_set,
                         infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):

    train_target, train_regressor = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressor = sm.add_constant(train_regressor)
    predictor = logistic_predict(train_target, train_regressor)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in
                           test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_data]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    print (error_rate, errors, len(test_predictions), predictor)
    return (error_rate, errors, len(test_predictions), predictor)


# def elapsed_timer():
#     start = default_timer()
#     elapser = lambda: default_timer() - start
#     yield lambda: elapser()
#     end = default_timer()
#     elapser = lambda: end - start
#
#
# def logistic_predictor_from_data(train_targets, train_regressors):
#     logit = sm.Logit(train_targets, train_regressors)
#     predictor = logit.fit(disp=0)
#     # print(predictor.summary())
#     return predictor
#
#
# def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1,
#                          infer_subsample=0.1):
#     """Report error rate on test_doc sentiments, using supplied model and train_docs"""
#
#     train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
#     train_regressors = sm.add_constant(train_regressors)
#     predictor = logistic_predictor_from_data(train_targets, train_regressors)
#
#     test_data = test_set
#     if infer:
#         if infer_subsample < 1.0:
#             test_data = sample(test_data, int(infer_subsample * len(test_data)))
#         test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in
#                            test_data]
#     else:
#         test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
#     test_regressors = sm.add_constant(test_regressors)
#
#     # Predict & evaluate
#     test_predictions = predictor.predict(test_regressors)
#     corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
#     errors = len(test_predictions) - corrects
#     error_rate = float(errors) / len(test_predictions)
#     return (error_rate, errors, len(test_predictions), predictor)


if __name__ == "__main__":
    # count num of processor of cpu and then this would be faster

    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    stemmer = PorterStemmer()

    path = '/Users/liyaguan/Documents/CS_550/reviews_Musical_Instruments_5.json'

    df = sqlContext.read.json(path).rdd

    df.take(5)

    stop_words = set(stopwords.words('english'))

    print(df.first())

    print(df.first()['reviewText'])

    text = df.map(lambda row: row['reviewerID'])
    text_user = df.map(lambda row: (row['reviewText'],row['reviewerID']))
    user = df.map(lambda row: row['reviewerID'])

    num_lines = text.countByKey()

    #index = sc.parallelize(xrange(num_lines))
    #print(index.take(5))

    # user-review map built by this line
    processed_text = text_user.map(process_text)
    # n = processed_text.count()
    processed_text_only = processed_text.map(lambda x: x[0])
    #n = len(processed_text_only.collect())
    processed_lable_only = processed_text.map(lambda x: x[1])
    #sentences = processed_text_only
    #thelables = text_user.map(process_lable)
    #sentences = {TaggedDocument(words=processed_text_only.collect(),tags=processed_lable_only.collect())}
    sentences = make_tag_doc(processed_text_only,processed_lable_only)[0]
    print(text.count())
    print(processed_text_only.take(5))
    # processed_text is the list of words of a review

    #test, train = processed_text.randomSplit(weights=[0.3,0.7], seed= 1234)

    #test = {TaggedDocument(words=test.map(lambda x: x[0]).collect(),tags=test.map(lambda x: x[1]).collect())}
    #train = {TaggedDocument(words=train.map(lambda x: x[0]).collect(),tags=train.map(lambda x: x[1]).collect())}

    #test = make_tag_doc(test.map(lambda x: x[0]),test.map(lambda x: x[1]),n)
    #train = make_tag_doc(train.map(lambda x: x[0]),train.map(lambda x: x[1]),n)

    #sentences = make_tag_doc(processed_text_only,processed_lable_only,n)[1]
    #sentences = make_tag_doc(processed_text_only,processed_lable_only,n)[2]

    #alldocs = processed_text_only.collect()
    #doc_list = alldocs

    #for i in e

    cores = multiprocessing.cpu_count()

    simple_models = [
        # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/ average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]

    #print(simple_models[1])
    #.toPandas().to_csv('hundred_all.csv')
    #.select('probability').toPandas().to_csv('output_index_36.csv')

    # Speed up setup by sharing results of the 1st model's vocabulary scan
    simple_models[1].build_vocab(sentences)  # PV-DM w/ concat requires one special NULL word so it serves as template
    #print(simple_models[0])


    ivec = simple_models[1].infer_vector(processed_text_only.collect(), alpha = 0.1, min_alpha=0.0001, steps=5)
    print(simple_models[1].most_similar(positive=[ivec], topn =10))
    print(simple_models[1].most_similar_cosmul(positive=[ivec],topn=10))


    for model in simple_models[1:]:
        model.reset_from(simple_models[0])
        print(model)






    models_by_name = OrderedDict((str(model), model) for model in simple_models)



    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    best_error = defaultdict(lambda: 1.0)

    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes

    print("START %s" % datetime.datetime.now())

    for epoch in range(passes):
        #shuffle(doc_list)  # Shuffling gets best results

        for name, train_model in models_by_name.items():
            # Train
            duration = 'na'
            train_model.alpha, train_model.min_alpha = alpha, alpha
            with elapsed_timer() as elapsed:
                train_model.train(doc_list, total_examples=len(doc_list), epochs=1)
                duration = '%.1f' % elapsed()

            # Evaluate
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs)
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if err <= best_error[name]:
                best_error[name] = err
                best_indicator = '*'
            print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

            if ((epoch + 1) % 5) == 0 or epoch == 0:
                eval_duration = ''
                with elapsed_timer() as eval_elapsed:
                    infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs,
                                                                                       test_docs, infer=True)
                eval_duration = '%.1f' % eval_elapsed()
                best_indicator = ' '
                if infer_err < best_error[name + '_inferred']:
                    best_error[name + '_inferred'] = infer_err
                    best_indicator = '*'
                print("%s%f : %i passes : %s %ss %ss" % (
                best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))

        print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

    print("END %s" % str(datetime.datetime.now()))

    # Print best error rates achieved
    print("Err rate Model")
    for rate, name in sorted((rate, name) for name, rate in best_error.items()):
        print("%f %s" % (rate, name))

    doc_id = np.random.randint(simple_models[0].docvecs.count)  # Pick random doc; re-run cell for more examples
    print('for doc %d...' % doc_id)
    for model in simple_models:
        inferred_docvec = model.infer_vector(alldocs[doc_id].words)
        print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))


    # ------------------------------------------
    #assert gensim.models.doc2vec.FAST_VERSION

    #model = Doc2Vec([sentences.collect()], size=100, negative=5, hs=0, min_count=2, window = 4, workers=cores)
    #model.build_vocab(sentences.collect())
    # print("START %s" % datetime.datetime.now())
    # alpha, min_alpha, passes = (0.025, 0.001, 20)
    # alpha_delta = (alpha - min_alpha)/passes

    # PV-DBOW
    # model = Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores)
    #
    # # Build vocab before tarining the model
    # model.build_vocab(sentences)
    #
    # name = "PV-DBOW"
    # best_error = defaultdict(lambda:1.0)
    #
    # for epoch in range(passes):
    #     #sentences = sentences.repartition(sentences.count())
    #     print ("iteration " + str(epoch+1))
    #
    #     # train
    #     duration = 'na'
    #     #with elapsed_timer() as elapsed:
    #         #trainning model
    #     model.train(sentences, total_examples=len(sentences), epochs=1)
    #     #    duration = '%.1f' % elapsed()
    #
    #     # evaluate
    #     eval_duration = 'na'
    #     #with elapsed_timer() as eval_elapsed:
    #     err, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
    #     #eval_duration = '%.1f' % eval_elapsed()
    #
    #     best_indicator = ' '
    #     if err <= best_error:
    #         best_error = err
    #         best_indicator = '*'
    #     print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1,name , duration, eval_duration))
    #
    #     if ((epoch + 1) % 5) == 0 or epoch == 0:
    #         eval_duration = ''
    #         with elapsed_timer() as eval_elapsed:
    #             infer_err, err_count, test_count, predictor = error_rate_for_model(model, train, test,
    #                                                                                infer=True)
    #         eval_duration = '%.1f' % eval_elapsed()
    #         best_indicator = ' '
    #         if infer_err < best_error[name + '_inferred']:
    #             best_error[name + '_inferred'] = infer_err
    #             best_indicator = '*'
    #         print("%s%f : %i passes : %s %ss %ss" % (
    #         best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))
    #
    #     print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
    #     alpha -= alpha_delta
    # print("END %s" % str(datetime.datetime.now()))


    # evaluate
    #err, err_count, test_count, predictor  = error_rate_for_model(model, train.collect(), test.collect())


    # wordtovec(processed_text_only)

    # the doc2vec not word to vec
    # build the doc2vec aka PV-DBOW model


    sc.stop()





#print(text.collect())

#print(df.collect())