import tensorflow as tf
import random
import numpy as np
import label_model

def _getRandom(examples,embeddings) :
  neg = random.randint(0,embeddings.getEmbeddings().shape[0]-1)
  for item_list in examples :
    if neg in item_list :
      return _getRandom( examples, embeddings )
  return neg

def _getScore(embeddings_in,category_in,model,sess) :
  score = sess.run(model.correct,feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:1.0})
  return score

def _doEpoch(embeddings_in,category_in,model,sess) :
  score = _getScore(embeddings_in,category_in,model,sess)
  sess.run(        model.train,  feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:.5})
  return score

def _meanExamples(examples) :
  return int(round( 1. * _totalExamples(examples) / len(examples) ))

def _totalExamples(examples) :
  return sum([ len(item) for item in examples ])

def _prepareDataForTraining(examples,embeddings,sample_size=0,has_unknown=False,include_unknown=True) :
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
    unknown_examples = [ _getRandom(examples,embeddings) for item in range(sample_size)]
    example_category.extend( [ identity[-1,:] for subitem in unknown_examples ] )
    example_embeddings.extend( [ embeddings.getEmbeddings()[ subitem ] for subitem in unknown_examples ] )

  return example_embeddings, example_category

def _splitTrainingAndTest(examples,split_fraction =.5) :
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

def _doTraining(examples,training_examples,test_examples,embeddings,has_unknown=False) :
  label_graph = tf.Graph()
  with label_graph.as_default() :
    model = label_model.model(number_of_classes=len(examples) + (1 if has_unknown else 0) )
    with tf.Session(graph=label_graph) as sess :
      sess.run( tf.global_variables_initializer() )
      best_correct = 0
      sample_size = _meanExamples(examples)
      total_examples = _totalExamples(examples)
      print "Training epoch,train,test ",
      for epoch_count in range(100) :

        # training
        example_embeddings, example_category = _prepareDataForTraining(training_examples,embeddings,sample_size,has_unknown=has_unknown)
        result_correct = _doEpoch( example_embeddings, example_category, model, sess )

        # test it!
        example_embeddings, example_category = _prepareDataForTraining(test_examples,embeddings,has_unknown=has_unknown,include_unknown=False)
        test_correct = _getScore( example_embeddings, example_category, model, sess )

        if epoch_count > 0 and epoch_count % 10 == 0 :
          print ": %3d,%.3f,%.3f" % ( epoch_count, 1-result_correct, 1-test_correct ),

        best_correct = max( best_correct , test_correct )

        if total_examples >= 20 and epoch_count >= 10 and ((1-test_correct) >= (1-best_correct) * 1.5) :
          print "End > test error",
          break
        if total_examples >= 20 and epoch_count >= 10 and 1-result_correct < 0.0001 :
          print "End > train fit ",
          break

      print ": Final %3d,%.3f,%.3f" % ( epoch_count, 1-result_correct, 1-test_correct )
      
      return model,sess.run(
        model.category_out,
        feed_dict={
          model.emb_in:embeddings.getEmbeddings(),
          model.dropout:1.0
          }
        )


def predictiveMultiClassWeights(examples,embeddings) :
  has_unknown = True if len(examples) < 3 else False

  subset_A , subset_B = _splitTrainingAndTest(examples)
  model,weights_A = _doTraining(examples,subset_A,subset_B,embeddings,has_unknown)
  model,weights_B = _doTraining(examples,subset_B,subset_A,embeddings,has_unknown)
  weights = ( weights_A + weights_B ) / 2.

  return weights
