
class Embeddings:
  def __init__(self, predictor=None, model=None, max_search_size=2000) :
    self.data_set = []
    self.all_embeddings = []
    self.max_search_size = max_search_size
    if predictor == None :
      import autoencode_predict
      autoencode_predict.restore()
      self.autoencode_predict = autoencode_predict
    else :
      self.autoencode_predict = predictor
    if model == None :
      self.model = self.autoencode_predict.autoencode_model
    else :
      self.model = model

  def reset(self) :
    self.all_embeddings = []

  def getEmbeddings(self) :
    if len(self.all_embeddings) == 0 :
      self.all_embeddings = self.autoencode_predict.sess.run(self.model.embedding,
        feed_dict={self.model.x_in:self.data_set} ).reshape([-1,self.model.SIZE])
    return self.all_embeddings
