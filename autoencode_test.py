import autoencode_predict
from data_source import LazyLoadWrapper, BatchWrapper, ResizeWrapper, ReshapeWrapper, Mnist

import unittest
 
class TestAutoEncodePredict(unittest.TestCase):
 
    def setUp(self):
        self.predictor = autoencode_predict.predict()
        self.predictor.restore()

    def tearDown(self):
        self.predictor.save()
 
    def test_doEpoch(self):
        autoencode_model = self.predictor.autoencode_model
        data = LazyLoadWrapper(BatchWrapper(ResizeWrapper(ReshapeWrapper(Mnist(), [28, 28, 1]), [32, 32])))
        result = self.predictor.doEpochOfTraining(autoencode_model.loss_5, autoencode_model.train_5, data_feed=data, batches=10, batch_size=50, elapse=10)
        self.assertLess( result[-1] , result[0] )


 
if __name__ == '__main__':
    unittest.main()
