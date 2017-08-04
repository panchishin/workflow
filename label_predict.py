import tensorflow as tf
import random
import numpy as np
import label_model




def getRandom(examples,embeddings) :
  neg = random.randint(0,embeddings.getEmbeddings().shape[0]-1)
  for item_list in examples :
    if neg in item_list :
      return getRandom( examples, embeddings )
  return neg


def getScore(embeddings_in,category_in,model,sess) :
  score = sess.run(model.correct,feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:1.0})
  return score

def doEpoch(embeddings_in,category_in,model,sess) :
  score = getScore(embeddings_in,category_in,model,sess)
  sess.run(        model.train,  feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:.5})
  return score

def meanExamples(examples) :
  return int(round( 1. * totalExamples(examples) / len(examples) ))

def totalExamples(examples) :
  return sum([ len(item) for item in examples ])

def prepareDataForTraining(examples,embeddings,sample_size=0,has_unknown=False,include_unknown=True) :
  example_category = []
  example_embeddings = []
  identity  = np.identity( len(examples) + (1 if has_unknown else 0) )
  for category in range(len(examples)) :
    if ( sample_size > 0 ) :
      mask = [] if len(examples[category]) == 0 else np.random.choice(len(examples[category]), sample_size)
    else :
      mask = range(len(examples[category]))
    example_category.extend( np.array([ identity[category,:] for item in examples[category] ])[mask] )
    example_embeddings.extend( np.array([ embeddings.getEmbeddings()[ subitem ] for subitem in examples[category] ])[mask] )

  if ( has_unknown and include_unknown ) :
    unknown_examples = [ getRandom(examples,embeddings) for item in range(sample_size)]
    example_category.extend( [ identity[-1,:] for subitem in unknown_examples ] )
    example_embeddings.extend( [ embeddings.getEmbeddings()[ subitem ] for subitem in unknown_examples ] )

  return example_embeddings, example_category

def splitTrainingAndTest(examples) :
  # split data into training vs test
  split_fraction = .2
  training_examples = []
  test_examples = []
  for grouping in examples :
    training_grouping = []
    test_grouping = []
    for item in grouping :
      if random.random() >= split_fraction :
        training_grouping.append( item )
      else :
        test_grouping.append( item )
    training_examples.append(training_grouping)
    test_examples.append(test_grouping)
  return training_examples , test_examples

def doTraining(examples,embeddings,has_unknown=False) :
  label_graph = tf.Graph()
  with label_graph.as_default() :
    model = label_model.model(number_of_classes=len(examples) + (1 if has_unknown else 0) )
    with tf.Session(graph=label_graph) as sess :
      sess.run( tf.global_variables_initializer() )
      print "Training started"
      best_correct = 0
      sample_size = meanExamples(examples)
      total_examples = totalExamples(examples)
      training_examples , test_examples = splitTrainingAndTest(examples)

      for epoch_count in range(200) :

        # training
        example_embeddings, example_category = prepareDataForTraining(training_examples,embeddings,sample_size,has_unknown=has_unknown)
        result_correct = doEpoch( example_embeddings, example_category, model, sess )

        # test it!
        example_embeddings, example_category = prepareDataForTraining(test_examples,embeddings,has_unknown=has_unknown,include_unknown=False)
        test_correct = getScore( example_embeddings, example_category, model, sess )

        if epoch_count % 20 == 0 :
          print "We got some training",int(round((1-result_correct)*1000))," and test",int(round((1-test_correct)*1000)),"results"

        best_correct = max( best_correct , test_correct )

        if total_examples >= 20 and epoch_count > 10 and ((1-test_correct) >= (1-best_correct) * 2) :
          print "End condition, test errors doubled from best test errors"
          break

      print "done training with training",int(round((1-result_correct)*1000))," and test",int(round((1-test_correct)*1000)),"results"
      return model,sess.run(
        model.category_out,
        feed_dict={
          model.emb_in:embeddings.getEmbeddings(),
          model.dropout:1.0
          }
        )


def predictiveMultiClassWeights(examples,embeddings) :
  has_unknown = True if len(examples) < 3 else False
  model,weights = doTraining(examples,embeddings,has_unknown)
  identity  = np.identity( len(examples) + (1 if has_unknown else 0) )
  for category in range(len(examples)) :
    for example in examples[category] :
      weights[example] = identity[category,:]
  return weights
