import session
import random
import numpy as np
import label_model


sess = session.sess


def getRandom(positive_examples,embeddings) :
  neg = positive_examples[0]
  while neg in positive_examples :
    neg = random.randint(0,embeddings.getEmbeddings().shape[0]-1)
  return neg

def doEpoch(positive_examples,embeddings) :
  score = 0.0
  for iteration in range(len(positive_examples)) :
    pos = positive_examples[ iteration ]
    neg = getRandom(positive_examples,embeddings)
    percent_correct = sess.run(label_model.correct,feed_dict={label_model.emb_in:embeddings.getEmbeddings()[ [pos,neg] ],label_model.category_in:[ [1.,0.],[0.,1.] ],label_model.dropout:1.0})
    score += percent_correct
    sess.run(label_model.train,feed_dict={label_model.emb_in:embeddings.getEmbeddings()[ [pos,neg] ],label_model.category_in:[ [1.,0.],[0.,1.] ],label_model.dropout:.5})
  score = 1. * score / len(positive_examples)
  print ".",
  return score

def doTraining(positive_examples,embeddings) :
  sess.run(label_model.init_new_vars_op)
  target_correct = max( 0.95 , min( 0.99 , 1. - 3. / len(positive_examples) ) )
  print "Target correct is",target_correct
  print "Training started",
  result_correct = 0
  for _ in range(30) :
    result_correct = doEpoch(positive_examples,embeddings)
    if result_correct >= target_correct :
      break
  print "done training with result correct :",result_correct      
  return sess.run(
    label_model.category_out,
    feed_dict={
      label_model.emb_in:embeddings.getEmbeddings(),
      label_model.dropout:1.0
      }
    )[:,0]

def getPredictiveWeights(positive_examples,negative_examples,embeddings) :
  weights = doTraining(positive_examples,embeddings)
  for element in positive_examples :
    weights[element] = 1.
  for element in negative_examples :
    weights[element] = 0.
  return weights


def groundTruthReport(positive_examples,embeddings,mnist) :
  category_in_full_set = np.array( [ [1.,0.] if item == 4 else [0.,1.] for item in np.argmax(mnist.test.labels,1) ] )

  data = sess.run(
    label_model.category_out,
    feed_dict={
      label_model.emb_in:embeddings.getEmbeddings(),
      label_model.category_in:category_in_full_set,
      label_model.dropout:1.0
      }
    )

  data = np.argmax( data, 1 )
  ground = np.argmax( category_in_full_set , 1 )

  print "Confusion Matrix"
  print "ground , prediction"
  print ( ( data == 0 ) * ( ground == 0 ) ).sum() , ( ( data == 1 ) * ( ground == 0 ) ).sum()
  print ( ( data == 0 ) * ( ground == 1 ) ).sum() , ( ( data == 1 ) * ( ground == 1 ) ).sum()

  precision = 1. * (( data == 0 ) * ( ground == 0 ) ).sum() / ( data == 0 ).sum()
  print "Precision", precision
  recall = 1. * (( data == 0 ) * ( ground == 0 ) ).sum() / ( ground == 0 ).sum()
  print "Recall",recall
  print "F1", 2.*precision*recall/(precision + recall)
