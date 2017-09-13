
class Embeddings:
  def __init__(self, session=None, model=None, max_search_size=2000) :
    self.data_set = []
    self.all_embeddings = []
    self.max_search_size = max_search_size
    if session == None :
      import session as importSession
      self.session = importSession
    else :
      self.session = session
    if model == None :
      self.model = self.session.autoencode_model
    else :
      self.model = model

  def reset(self) :
    self.all_embeddings = []

  def getEmbeddings(self) :
    if len(self.all_embeddings) == 0 :
      self.all_embeddings = self.session.sess.run(self.model.embedding,
        feed_dict={self.model.x_in:self.data_set} ).reshape([-1,self.model.SIZE])
    return self.all_embeddings
