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


def doEpoch(embeddings_in,category_in,model,sess) :
  score = sess.run(model.correct,feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:1.0})
  sess.run(        model.train,  feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:.5})
  return score


def doTraining(examples,embeddings) :
  label_graph = tf.Graph()
  with label_graph.as_default() :
    model = label_model.model(number_of_classes=1+len(examples) )
    with tf.Session(graph=label_graph) as sess :
      sess.run( tf.global_variables_initializer() )
      target_correct = max( 0.95 , min( 0.998 , 1. - 5. / ( len(examples) * len(examples[0]) ) ) )
      print "Target correct is",target_correct
      print "Training started",
      result_correct = 0
      for epoch_count in range(50) :

        example_category = []
        example_embeddings = []
        identity  = np.identity( len(examples) + 1 )
        for category in range(len(examples)) :
          example_category.extend( [ identity[category,:] for item in examples[category] ] )
          example_embeddings.extend( [ embeddings.getEmbeddings()[ subitem ] for subitem in examples[category] ] )

        negative_examples = [ getRandom(examples,embeddings) for item in examples[0]]
        example_category.extend( [ identity[-1,:] for subitem in negative_examples ] )
        example_embeddings.extend( [ embeddings.getEmbeddings()[ subitem ] for subitem in negative_examples ] )

        result_correct = doEpoch( example_embeddings, example_category, model, sess )

        if epoch_count % 10 == 0 :
          print int(round(result_correct*100)),

        if result_correct >= target_correct :
          break

      print "done training with result correct :",result_correct      
      return model,sess.run(
        model.category_out,
        feed_dict={
          model.emb_in:embeddings.getEmbeddings(),
          model.dropout:1.0
          }
        )


def predictiveBinaryWeights(positive_examples,negative_examples,embeddings) :
  model,weights = doTraining([positive_examples],embeddings)
  for element in positive_examples :
    weights[element] = [1.,0.]
  for element in negative_examples :
    weights[element] = [0.,1.]
  return weights


def predictiveMultiClassWeights(examples,embeddings) :
  model,weights = doTraining(examples,embeddings)
  identity  = np.identity( len(examples) + 1 )
  for category in range(len(examples)) :
    for example in examples[category] :
      weights[example] = identity[category,:]
  return weights
