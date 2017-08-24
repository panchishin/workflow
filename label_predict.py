import tensorflow as tf
import random
import numpy as np
import label_model

def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()


def _getRandom(examples,embeddings) :
  neg = random.randint(0,embeddings.getEmbeddings().shape[0]-1)
  for item_list in examples :
    if neg in item_list :
      return _getRandom( examples, embeddings )
  return neg


def _getScore(embeddings_in,category_in,model,sess) :
  score = sess.run(model.correct,feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:1.0})
  return score

def _doEpoch(embeddings_in,category_in,weight_in,model,sess) :
  score = _getScore(embeddings_in,category_in,model,sess)
  sess.run( model.train, feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:.5,model.weight_in:weight_in})
  return score

def _meanExamples(examples) :
  return int(round( 1. * _totalExamples(examples) / len(examples) ))

def _totalExamples(examples) :
  return sum([ len(item) for item in examples ])

def _prepareDataForTensorflow(examples,embeddings,importance={},sample_size=0,has_unknown=False,include_unknown=True) :
  example_weight = []
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
    example_weight.extend( [ importance.get(item,1.) for item in np.array(examples[category])[mask] ] )

  if ( has_unknown and include_unknown ) :
    unknown_examples = [ _getRandom(examples,embeddings) for item in range(sample_size)]
    example_category.extend( [ identity[-1,:] for subitem in unknown_examples ] )
    example_embeddings.extend( [ embeddings.getEmbeddings()[ subitem ] for subitem in unknown_examples ] )
    example_weight.extend( [ importance.get(item,1.) for item in unknown_examples ] )

  return example_embeddings, example_category, example_weight

def _splitTrainingAndTest(examples, split_fraction=.5) :
  train_examples = []
  test_examples = []
  for grouping in examples :
    training_grouping = []
    test_grouping = []
    for item in grouping :
      if random.random() >= split_fraction :
        training_grouping.append( item )
      else :
        test_grouping.append( item )
    train_examples.append(training_grouping)
    test_examples.append(test_grouping)
  return train_examples , test_examples

def _doTraining(examples, train_examples, test_examples, embeddings, importance, has_unknown=False) :
  label_graph = tf.Graph()
  with label_graph.as_default() :
    model = label_model.model(number_of_classes=len(examples) + (1 if has_unknown else 0) )
    with tf.Session(graph=label_graph) as sess :
      sess.run( tf.global_variables_initializer() )
      print "Training started"
      best_correct = 0
      sample_size = _meanExamples(examples)
      total_examples = _totalExamples(examples)
      train_examples , test_examples = _splitTrainingAndTest(examples)

      for epoch_count in range(150) :

        # training
        example_embeddings, example_category, example_weight = _prepareDataForTensorflow(train_examples,embeddings,importance,has_unknown=has_unknown)
        weight_in = np.ones(len(example_category))
        result_correct = _doEpoch( example_embeddings, example_category, weight_in, model, sess )

        # test it!
        example_embeddings, example_category, example_weight = _prepareDataForTensorflow(test_examples,embeddings,importance,has_unknown=has_unknown,include_unknown=False)
        test_correct = _getScore( example_embeddings, example_category, model, sess )

        if epoch_count % 20 == 0 :
          print "epoch %3d : %.4f training and %0.4f test error" % ( epoch_count, (1-result_correct),(1-test_correct) )

        best_correct = max( best_correct , test_correct )

        if total_examples >= 10 and epoch_count > 10 and ((1-test_correct) >= (1-best_correct) * 1.2) :
          print "End condition, test errors doubled from best test errors"
          break

      print "done training with %.4f training and %0.4f test error" % ( (1-result_correct),(1-test_correct) )
      return model,sess.run( model.category_out, feed_dict={ model.emb_in:embeddings.getEmbeddings(), model.dropout:1.0 } )


def _has_unknown(examples) :
  return True if len(examples) < 3 else False

def _knownWeightsToGroundTruth(examples,weights) :
  identity  = np.identity( len(examples) + (1 if _has_unknown(examples) else 0) )
  for category in range(len(examples)) :
    for example in examples[category] :
      weights[example] = identity[category,:]
  return weights




def groupingsToLabelsAndWeights(groupings,weights) :
  labels = []
  result_weights = []
  result_category = []
  identity  = np.identity( len(groupings) + (1 if _has_unknown(groupings) else 0) )
  for category in range(len(groupings)) :
    for example in groupings[category] :
      labels.append( example )
      result_weights.append( weights[example] )
      result_category.append( category )
  return np.array(labels), np.array(result_weights), np.array(result_category)

def populateImportance(labels,values,result={}) :
  for index in range(labels.shape[0]) :
    result[ labels[index] ] = values[index]
  return result

def calculateImportanceFromTraining(example_split_a,weights_a,example_split_b,weights_b) :
  labels_a, result_weights_a, category_a = groupingsToLabelsAndWeights(example_split_a,weights_b)
  importance_a = -np.log( result_weights_a[np.arange(len(result_weights_a)), category_a ] )
  labels_b, result_weights_b, category_b = groupingsToLabelsAndWeights(example_split_b,weights_a)
  importance_b = -np.log( result_weights_b[np.arange(len(result_weights_b)), category_b ] )
  importance = populateImportance(labels_a,importance_a,populateImportance(labels_b,importance_b))
  return importance

def _calculateImportance(examples,embeddings,importance={}) :
  example_split_a , example_split_b = _splitTrainingAndTest(examples)
  model,weights_a = _doTraining( examples, example_split_a, example_split_b, embeddings, importance, _has_unknown(examples) )
  model,weights_b = _doTraining( examples, example_split_b, example_split_a, embeddings, importance, _has_unknown(examples) )
  print "== The average error over the whole dataset combined a+b is %0.4f ==" % ( (np.argmax( mnist.train.labels , 1 ) != np.argmax( weights_a + weights_b , 1 ) ).mean() )
  return calculateImportanceFromTraining(example_split_a,weights_a,example_split_b,weights_b)

def repeatcalculateImportance(examples,embeddings) :
  lambda_importance = {}
  lambda_count = 1

  for trials in range(5) :
    importance = _calculateImportance(examples,embeddings,lambda_importance)
    for element in importance.keys() :
      lambda_importance[element] = lambda_importance.get(element,1.) * ( 1.0 - 1.0 / lambda_count ) + importance.get(element,1.) * ( 1.0 / lambda_count )
    lambda_count += 1
  return lambda_importance

def predictiveMultiClassWeights(examples,embeddings) :
  importance = repeatcalculateImportance(examples,embeddings)
  example_split_a , example_split_b = _splitTrainingAndTest(examples)
  print "First Training"
  model,weights_a = _doTraining( examples, example_split_a, example_split_b, embeddings, importance, _has_unknown(examples) )
  print "Second Training"
  model,weights_b = _doTraining( examples, example_split_b, example_split_a, embeddings, importance, _has_unknown(examples) )
  return _knownWeightsToGroundTruth(examples, (weights_a+weights_b)/2 )

