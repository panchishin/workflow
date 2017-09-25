class Embeddings:

    def __init__(self, predictor=None, max_search_size=2000):
        self.data_set = []
        self.all_embeddings = []
        self.max_search_size = max_search_size
        if predictor is None:
            import autoencode_predict
            autoencode_predict.restore()
            self.autoencode_predict = autoencode_predict
        else:
            self.autoencode_predict = predictor
        self.autoencode_model = self.autoencode_predict.autoencode_model

    def reset(self):
        self.all_embeddings = []

    def getEmbeddings(self):
        if len(self.all_embeddings) == 0:
            feed_dict = {self.autoencode_model.x_in: self.data_set}

            self.all_embeddings = self.autoencode_predict.sess.run(self.autoencode_model.embedding, feed_dict=feed_dict)
            self.all_embeddings = self.all_embeddings.reshape([self.all_embeddings.shape[0], -1])

        return self.all_embeddings
