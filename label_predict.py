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

def doEpoch(positive_examples,embeddings,model) :
  score = 0.0
  for iteration in range(len(positive_examples)) :
    pos = positive_examples[ iteration ]
    neg = getRandom(positive_examples,embeddings)
    percent_correct = sess.run(model.correct,feed_dict={model.emb_in:embeddings.getEmbeddings()[ [pos,neg] ],model.category_in:[ [1.,0.],[0.,1.] ],model.dropout:1.0})
    score += percent_correct
    sess.run(model.train,feed_dict={model.emb_in:embeddings.getEmbeddings()[ [pos,neg] ],model.category_in:[ [1.,0.],[0.,1.] ],model.dropout:.5})
  score = 1. * score / len(positive_examples)
  print ".",
  return score

model = None
def getModel() :
  global model
  if model == None :
    model = label_model.model()
  return model

def doTraining(positive_examples,embeddings,model=None) :
  model = getModel()
  sess.run(model.init_new_vars_op)
  target_correct = max( 0.95 , min( 0.99 , 1. - 3. / len(positive_examples) ) )
  print "Target correct is",target_correct
  print "Training started",
  result_correct = 0
  for _ in range(30) :
    result_correct = doEpoch(positive_examples,embeddings,model)
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

