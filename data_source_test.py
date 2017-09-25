import data_source

print "RUNNING TESTS"

files = ["166261_l.jpg", "165176_l.jpg", "164425_l.jpg"]
labels = ["bob", "betty", "bob"]
fileLoader = data_source.LazyLoadWrapper(data_source.FileReader(files, labels))
if fileLoader.getImages().shape == tuple([6, 240, 240, 3]):
    print ".",
else:
    print "FAIL", fileLoader.getImages().shape, "should equal [6,240,240,3]"

if fileLoader.getLabels().shape == tuple([6, 2]):
    print ".",
else:
    print "FAIL", fileLoader.getLabels().shape, "should equal [6,2]"


mnist = data_source.LazyLoadWrapper(data_source.Mnist())
if mnist.getImages().shape == tuple([55000, 28 * 28]):
    print ".",
else:
    print "FAIL", mnist.getImages().shape, "should equal [55000,28*28]"

reshaped = data_source.LazyLoadWrapper(data_source.ReshapeWrapper(mnist, [28, 28, 1]))
if reshaped.getImages().shape == tuple([55000, 28, 28, 1]):
    print ".",
else:
    print "FAIL", reshaped.getImages().shape, "should equal [55000,28,28,1]"

resized = data_source.LazyLoadWrapper(data_source.ResizeWrapper(reshaped, [32, 32]))
if resized.getImages().shape == tuple([55000, 32, 32, 1]):
    print ".",
else:
    print "FAIL", resized.getImages().shape, "should equal [55000,32,32,1]"

batch = data_source.BatchWrapper(resized)
batch.init()
if batch.nextBatch(10).shape == tuple([10, 32, 32, 1]):
    print ".",
else:
    print "FAIL", resized.getImages().shape, "should equal [10,32,32,1]"

print "done"
