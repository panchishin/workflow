import autoencode_predict
from data_source import LazyLoadWrapper, BatchWrapper, ResizeWrapper, ReshapeWrapper, Mnist

predictor = autoencode_predict.predict()

predictor.restore()
autoencode_model = predictor.autoencode_model

data = LazyLoadWrapper(BatchWrapper(ResizeWrapper(ReshapeWrapper(Mnist(), [28, 28, 1]), [32, 32])))

print "Start training test ..."
predictor.doEpochOfTraining(autoencode_model.loss_6, autoencode_model.train_6, data_feed=data, batches=10, batch_size=50)
print "... finished training test."
predictor.save()
