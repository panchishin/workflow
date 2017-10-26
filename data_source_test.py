import data_source
import unittest


class TestFileReader(unittest.TestCase):

    def setUp(self):
        files = ["166261_l.jpg", "165176_l.jpg", "164425_l.jpg"]
        labels = ["bob", "betty", "bob"]
        self.fileLoader = data_source.LazyLoadWrapper(data_source.FileReader(files, labels))

    def test_fileLoader_images(self):
        self.assertEqual( self.fileLoader.getImages().shape , tuple([6, 240, 240, 3]) )

    def test_fileLoader_label(self):
        self.assertEqual( self.fileLoader.getLabels().shape , tuple([6, 2]) )

    def test_concat(self):
        concat = data_source.LazyLoadWrapper(data_source.ConcatWrapper([self.fileLoader, self.fileLoader]))
        self.assertEqual( concat.getImages().shape , tuple([12, 240, 240, 3]) )

    def test_slice(self):
        slicer = data_source.LazyLoadWrapper(data_source.SliceWrapper(self.fileLoader, width=24, stride=12))
        self.assertEqual( slicer.getImages().shape , tuple([6 * 19 * 19, 24, 24, 3]) )


class TestMnist(unittest.TestCase):
    def setUp(self):
        self.mnist = data_source.LazyLoadWrapper(data_source.Mnist())
        self.reshaped = data_source.LazyLoadWrapper(data_source.ReshapeWrapper(self.mnist, [28, 28, 1]))
        self.resized = data_source.LazyLoadWrapper(data_source.ResizeWrapper(self.reshaped, [32, 32]))
 
    def test_check_image_shape(self):
        self.assertEqual( self.mnist.getImages().shape , tuple([55000, 28 * 28]) )

    def test_reshape(self):
        self.assertEqual( self.reshaped.getImages().shape , tuple([55000, 28, 28, 1]) )

    def test_resize(self):
        self.assertEqual( self.resized.getImages().shape , tuple([55000, 32, 32, 1]) )

    def test_batch(self):
        batch = data_source.BatchWrapper(self.resized)
        batch.init()
        self.assertEqual( batch.nextBatch(10).shape , tuple([10, 32, 32, 1]) )


 
if __name__ == '__main__':
    unittest.main()
