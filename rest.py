import autoencode_predict
import falcon
import os.path  # for serving html files
import json
import numpy as np
from embeddings import Embeddings
import nearest_neighbour
import label_predict
from sklearn.manifold import TSNE
from data_source import ConcatWrapper, SliceWrapper, LazyLoadWrapper, BatchWrapper, ResizeWrapper, ReshapeWrapper, Mnist, FileReader
import scipy.misc
import tempfile
import dummy_predict


predictor = dummy_predict


class Dataset:

    def __init__(self):
        self.value = None
        self.imageData = None
        self.dataSize = 0
        self.data_sets = {}

    def choose_mnist(self):
        print "CHOSE MNIST"
        global predictor, autoencode_model, embeddings

        predictor = autoencode_predict.predict(name="meta-data/mnist/autoencode_model", color_depth=1)
        predictor.stop()
        predictor.restore()
        autoencode_model = predictor.autoencode_model
        embeddings = Embeddings(predictor)

        print "Loading images ..."
        if 'mnist' not in self.data_sets:
            print "Key missing.  Building ImageData"
            imageData = LazyLoadWrapper(BatchWrapper(ResizeWrapper(ReshapeWrapper(Mnist(), [28, 28, 1]), [32, 32])))
            imageData.getImages()
            self.data_sets['mnist'] = imageData

        print "  mnist shape is", self.data_sets['mnist'].getImages().shape
        print "... loading images done"
        embeddings.data_set = self.data_sets['mnist'].getImages()
        return self.data_sets['mnist']

    def choose_garden(self):
        print "CHOSE GARDEN"
        global predictor, autoencode_model, embeddings

        predictor = autoencode_predict.predict(name="meta-data/garden/garden_model", color_depth=3)
        predictor.stop()
        predictor.restore()
        autoencode_model = predictor.autoencode_model
        embeddings = Embeddings(predictor)
        config_data = json.load(open("data/file_data.json", "r"))

        print "Loading images ..."
        if 'garden' not in self.data_sets:
            print "Key missing.  Building ImageData"

            print "Loading files ...",
            files = LazyLoadWrapper(ResizeWrapper(FileReader(config_data["file_names"], config_data["labels"]), [64, 64]))
            files.init()
            print "done."
            print "Calculating full size ...",
            full_size = LazyLoadWrapper(ResizeWrapper(files, [32, 32]))
            full_size.init()
            print "done."
            print "Calculating half size ...",
            half_size = LazyLoadWrapper(SliceWrapper(files, 32, 16))
            half_size.init()
            print "done."
            print "Calculating concat the whole thing ...",
            self.data_sets['garden'] = LazyLoadWrapper(BatchWrapper(ConcatWrapper([full_size, half_size])))
            print "done."

            self.data_sets['garden'].getImages()

        print "  garden shape is", self.data_sets['garden'].getImages().shape
        print "... loading images done"
        embeddings.data_set = self.data_sets['garden'].getImages()
        return self.data_sets['garden']

    def on_get(self, req, resp):
        resp.body = json.dumps({'response': {'result': 'success', 'value': self.value,
                                             'size': self.dataSize, 'available': ['mnist', 'garden']}})

    def on_post(self, req, resp):
        data_text = req.stream.read()
        data = json.loads(data_text)

        if data['value'] == "mnist" and data['value'] != self.value:
            self.imageData = self.choose_mnist()
        if data['value'] == "garden" and data['value'] != self.value:
            self.imageData = self.choose_garden()

        self.value = data['value']
        self.dataSize = self.imageData.getImages().shape
        resp.body = json.dumps({'response': {'result': 'success', 'set': self.value, 'size': self.imageData.getImages().shape}})


dataset = Dataset()


def falconRespondArrayAsImage(data, resp):
    resp.content_type = 'image/png'
    with tempfile.TemporaryFile() as fp:
        scipy.misc.toimage(data).save(fp=fp, format="PNG")
        fp.seek(0)
        resp.body = fp.read()


class Display:

    def on_get(self, req, resp, file_name):
        if not os.path.isfile("view/" + file_name):
            return

        result = open("view/" + file_name, "r")
        if ("html" in file_name):
            resp.content_type = "text/html"
        else:
            resp.content_type = "text/plain"

        resp.body = result.read()
        result.close()


class LayerImage:

    def getExample(self, index, layer):
        image = dataset.imageData.getImages()[index:index + 1]
        data = predictor.sess.run(layer, feed_dict={autoencode_model.x_in: image})

        if image.shape[3] == 1:
            return data.reshape([autoencode_model.SIZE, autoencode_model.SIZE])
        else:
            return data.reshape(image.shape[1:])

    def on_get(self, req, resp, layer, index):
        try:
            ml_layer = [autoencode_model.x_noisy, autoencode_model.x_out_1, autoencode_model.x_out_2, autoencode_model.x_out_3,
                        autoencode_model.x_out_4, autoencode_model.x_out_5, autoencode_model.x_in][int(layer)]
            falconRespondArrayAsImage(
                self.getExample(int(index), ml_layer),
                resp
            )
        except Exception as e:
            print(e)


class DoLearning:

    def on_get(self, req, resp, index):
        print "Training Layer", index
        loss = [autoencode_model.loss_1, autoencode_model.loss_2, autoencode_model.loss_3, autoencode_model.loss_4,
                autoencode_model.loss_5, autoencode_model.loss_6][int(index)]
        train = [autoencode_model.train_1, autoencode_model.train_2, autoencode_model.train_3, autoencode_model.train_4,
                 autoencode_model.train_5, autoencode_model.train_6][int(index)]
        predictor.doEpochOfTraining(loss, train, dataset.imageData)
        embeddings.reset()
        resp.body = json.dumps({'response': 'done'})


class SessionControl:

    def on_get(self, req, resp, action):
        if action == 'reset':
            predictor.reset()
            embeddings.reset()
            resp.body = json.dumps({'response': 'done'})
            return
        if action == 'restore':
            predictor.restore()
            resp.body = json.dumps({'response': 'done'})
            return
        if action == 'save':
            predictor.save()
            resp.body = json.dumps({'response': 'done'})
            return

        resp.body = json.dumps({'response': 'valid actions are reset, restore, or save'})


class Similar:

    def on_get(self, req, resp, index):
        names = nearest_neighbour.byIndex(int(index), embeddings.getEmbeddings()).tolist()
        resp.body = json.dumps({'response': names})


class GroupPredict:

    def __init__(self):
        self.previous_group_predict_result = None
        self.previous_group_predict_data_hash = 0
        self.previous_group_predict_page = 0
        self.previous_response_index = -1

    def report(self, result):
        print "class :",
        for b in range(10):
            print "%5d" % b,
        print " "

        print "count :",
        predict = np.argmax(result, 1)
        prediction_sums = [(predict == b).sum() for b in range(10)]
        for item in prediction_sums:
            print "%5d" % item,
        rms = (sum([(item - sum(prediction_sums) / 10.) ** 2 for item in prediction_sums]) / 10.) ** .5
        print "RMS %5d" % rms

        print "  F1  :",
        total_f1 = 0.
        ground = np.argmax(dataset.imageData.getLabels(), 1)
        for a in range(10):
            precision = 1. * ((ground == a) * (predict == a)).sum() / (predict == a).sum()
            recall = 1. * ((ground == a) * (predict == a)).sum() / (ground == a).sum()
            f1 = (2. * precision * recall / (precision + recall + 0.01))
            total_f1 += f1
            print "%5d" % (100. * f1),
        print "AVG %5d" % (10. * total_f1)
        print "== Ground truth error %5.2f == error,top_percent,count : " % (
            100. * (np.argmax(dataset.imageData.getLabels(), 1) != np.argmax(result, 1)).mean()),
        for confidence in [1., .99, .90, .75, .5, .25, .1, .01]:
            conf_filter = np.argsort(np.max(result, 1))[::-1]
            conf_filter = conf_filter[: int(conf_filter.shape[0] * confidence)]
            print "%5.2f%s %4.2f %5d ," % (
                100. * (np.argmax(dataset.imageData.getLabels()[conf_filter, :], 1) != np.argmax(result[conf_filter, :], 1)).mean(),
                "%", confidence, conf_filter.shape[0]),
        print "\n"

    def on_post(self, req, resp, response_index):
        response_index = int(response_index)

        data_text = req.stream.read()
        data = json.loads(data_text)
        grouping = data['grouping']

        if hash(json.dumps(grouping)) == self.previous_group_predict_data_hash:
            result = self.previous_group_predict_result
            if response_index == self.previous_response_index:
                self.previous_group_predict_page = (self.previous_group_predict_page + 1) % 5
            else:
                self.previous_response_index = response_index
                self.previous_group_predict_page = 0
        else:
            print "The size of the embeddings is", embeddings.getEmbeddings().shape
            result, error = label_predict.predictiveMultiClassWeights(grouping, embeddings)
            self.previous_group_predict_result = result
            self.previous_group_predict_page = 0
            self.previous_group_predict_data_hash = hash(json.dumps(grouping))
            self.previous_error = error
            print "the result shape is", result.shape
            if (result.shape[1] == 10):
                self.report(result)

        max_result = np.max(result, 1)
        if response_index >= 0:
            result = np.array(result)[:, response_index]
        else:
            result = max_result

        isLabeled = np.zeros(result.shape[0])
        for category in range(len(grouping)):
            if category == response_index:
                for example in grouping[category]:
                    isLabeled[example] = 1

        if data['isLabeled'] == 1:
            likely_filter = (isLabeled >= 1.)
        else:
            likely_filter = (isLabeled < 1.) * (result == max_result)

        likely_weight = result[likely_filter]
        likely_index = np.array(range(result.shape[0]))[likely_filter]
        positive = likely_index[np.argsort(likely_weight)]

        if data['order'] == 'forward':
            positive = positive[::-1]

        index = int(positive.shape[0] * data['index'])
        positive = positive[index:]

        positive = positive[self.previous_group_predict_page * 10:][:10].tolist()

        resp.body = json.dumps({'response': {'positive': positive, 'error': self.previous_error.tolist()}})


class TSne:

    def calculate_tsne(self):
        print "Starting the TSNE calculation ..."
        self.max_size = 2500
        positions = TSNE(n_components=2).fit_transform(embeddings.getEmbeddings()[:self.max_size])
        positions = positions - np.min(positions, 0)
        positions = positions / np.max(positions, 0)
        self.positions = positions
        print "... finished the TSNE calculations."

    def __init__(self):
        self.primed = False

    def on_get(self, req, resp, size):
        if self.primed is False:
            self.primed = True
            self.calculate_tsne()
        size = min(int(size), self.max_size)
        data = []
        for index in range(size):
            data.append({'x': self.positions[index, 0], 'y': self.positions[index, 1], 'id': index})

        resp.body = json.dumps({'response': data})


"""
================================
Add the endpoints to the service
================================
"""
api = falcon.API()
api.add_route('/view/{file_name}', Display())
api.add_route('/layer{layer}/{index}', LayerImage())
api.add_route('/learn/{index}', DoLearning())
api.add_route('/session/{action}', SessionControl())
api.add_route('/similar/{index}', Similar())
api.add_route('/group_predict/{response_index}', GroupPredict())
api.add_route('/tsne/{size}', TSne())
api.add_route('/dataset', dataset)
