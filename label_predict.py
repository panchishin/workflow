import tensorflow as tf
import random
import numpy as np
import label_model




def getRandom(positive_examples,embeddings) :
  neg = positive_examples[0]
  while neg in positive_examples :
    neg = random.randint(0,embeddings.getEmbeddings().shape[0]-1)
  return neg

def doEpoch(examples,embeddings,model,sess) :
  score = 0
  for example in examples :
    embeddings_in = [ embeddings.getEmbeddings()[ subitem ] for subitem in example ]
    category_in = np.identity( len(example) )
    score += sess.run(model.correct,feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:1.0})
    sess.run(         model.train,  feed_dict={model.emb_in:embeddings_in,model.category_in:category_in,model.dropout:.5})
  score = 1. * score / len(examples)
  print ".",
  return score


def doTraining(positive_examples,embeddings) :
  label_graph = tf.Graph()
  with label_graph.as_default() :
    model = label_model.model(number_of_classes=2)
    with tf.Session(graph=label_graph) as sess :
      sess.run( tf.global_variables_initializer() )
      target_correct = max( 0.95 , min( 0.99 , 1. - 3. / len(positive_examples) ) )
      print "Target correct is",target_correct
      print "Training started",
      result_correct = 0
      for _ in range(30) :
        negative_examples = [ getRandom(positive_examples,embeddings) for item in positive_examples]
        result_correct = doEpoch( zip( positive_examples , negative_examples ) ,embeddings,model,sess)
        if result_correct >= target_correct :
          break
      print "done training with result correct :",result_correct      
      return model,sess.run(
        model.category_out,
        feed_dict={
          model.emb_in:embeddings.getEmbeddings(),
          model.dropout:1.0
          }
        )[:,0]

def getPredictiveWeights(positive_examples,negative_examples,embeddings) :
  model,weights = doTraining(positive_examples,embeddings)
  for element in positive_examples :
    weights[element] = 1.
  for element in negative_examples :
    weights[element] = 0.
  return weights

