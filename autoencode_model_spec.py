import autoencode_model
import session

session.restoreSession()

from data_source import LazyLoadWrapper, BatchWrapper, ResizeWrapper, ReshapeWrapper, Mnist
data = LazyLoadWrapper( BatchWrapper( ResizeWrapper( ReshapeWrapper( Mnist(), [28,28,1] ) , [32,32] ) ) )

print "Start training test ..."
session.doEpochOfTraining( autoencode_model.loss_6 , autoencode_model.train_6 , data_feed=data, batches=10 , batch_size=50 )
print "... finished training test."
session.saveSession()


