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
        result = self.predictor.doEpochOfTraining(autoencode_model.loss_6, autoencode_model.train_6, data_feed=data, batches=10, batch_size=50)
        self.assertLess( result[0]['loss'] , -19)
        self.assertLess( result[1]['loss'] , -19)

 
if __name__ == '__main__':
    unittest.main()

