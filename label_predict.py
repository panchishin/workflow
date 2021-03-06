import tensorflow as tf
import random
import numpy as np
import label_model


def _getRandom(examples, embeddings):
    neg = random.randint(0, embeddings.getEmbeddings().shape[0] - 1)
    for item_list in examples:
        if neg in item_list:
            return _getRandom(examples, embeddings)
    return neg


def _getScore(embeddings_in, category_in, model, sess):
    error, predicted_category = sess.run([model.error, model.category_out],
                                         feed_dict={model.emb_in: embeddings_in, model.category_in: category_in,
                                                    model.dropout: 1.0})
    return error, predicted_category


def _doEpoch(embeddings_in, category_in, model, sess):
    error, predicted_category = _getScore(embeddings_in, category_in, model, sess)
    sess.run(model.train, feed_dict={model.emb_in: embeddings_in, model.category_in: category_in, model.dropout: .5})
    return error, predicted_category


def _meanExamples(examples):
    return int(round(1. * _totalExamples(examples) / len(examples)))


def _totalExamples(examples):
    return sum([len(item) for item in examples])


def _prepareDataForTraining(examples, embeddings, sample_size=0, has_unknown=False, include_unknown=True):
    example_category = []
    example_embeddings = []
    identity = np.identity(len(examples) + (1 if has_unknown else 0))
    for category in range(len(examples)):
        if (sample_size > 0):
            mask = [] if len(examples[category]) == 0 else np.random.choice(len(examples[category]), sample_size)
        else:
            mask = range(len(examples[category]))
        example_category.extend(np.array([identity[category, :] for item in examples[category]])[mask])
        example_embeddings.extend(
            np.array([embeddings.getEmbeddings()[subitem] for subitem in examples[category]])[mask])

    if (has_unknown and include_unknown):
        unknown_examples = [_getRandom(examples, embeddings) for item in range(sample_size)]
        example_category.extend([identity[-1, :] for subitem in unknown_examples])
        example_embeddings.extend([embeddings.getEmbeddings()[subitem] for subitem in unknown_examples])

    return example_embeddings, example_category


def _splitTrainingAndTest(examples, split_fraction=.5):
    training_examples = []
    test_examples = []
    for grouping in examples:
        training_grouping = []
        test_grouping = []
        for item in grouping:
            if random.random() >= split_fraction:
                training_grouping.append(item)
            else:
                test_grouping.append(item)
        training_examples.append(training_grouping)
        test_examples.append(test_grouping)
    return training_examples, test_examples


def _doTraining(examples, training_examples, test_examples, embeddings, has_unknown=False):

    label_graph = tf.Graph()
    with label_graph.as_default():
        model = label_model.model(number_of_classes=len(examples) + (1 if has_unknown else 0), width=embeddings.getEmbeddings().shape[1])
        with tf.Session(graph=label_graph) as sess:
            sess.run(tf.global_variables_initializer())
            best_error = 1.0
            sample_size = _meanExamples(examples)
            total_examples = _totalExamples(examples)
            print "Training epoch,train,test ",
            for epoch_count in range(50):

                # training
                example_embeddings, example_category = _prepareDataForTraining(training_examples, embeddings,
                                                                               sample_size, has_unknown=has_unknown)
                result_error, _ = _doEpoch(example_embeddings, example_category, model, sess)

                # test it!
                example_embeddings, example_category = _prepareDataForTraining(test_examples, embeddings,
                                                                               has_unknown=has_unknown,
                                                                               include_unknown=False)
                test_error, predicted_category = _getScore(example_embeddings, example_category, model, sess)

                if epoch_count > 0 and epoch_count % 10 == 0:
                    print ": %3d,%.3f,%.3f" % (epoch_count, result_error, test_error),

                best_error = min(best_error, test_error)

                if total_examples >= 20 and epoch_count >= 10 and (test_error >= best_error * 1.5):
                    print "End > test error",
                    break
                if total_examples >= 20 and epoch_count >= 10 and result_error < 0.0001:
                    print "End > train fit ",
                    break

            print ": Final %3d,%.3f,%.3f" % (epoch_count, result_error, test_error)

            example_category = np.array(example_category)
            highest_signal = np.argsort(np.max(predicted_category, 1))[::-1]
            fractions = [1.0]
            test_errors = [test_error]
            for fraction in [0.50, 0.25, 0.10, 0.05, 0.02, 0.01]:
                number = int(predicted_category.shape[0] * fraction)
                if number >= 100:
                    fractions.append(fraction)
                    highest_signal = highest_signal[:number]
                    errors = (np.argmax(example_category[highest_signal, :], 1) != np.argmax(
                        predicted_category[highest_signal, :], 1)).mean()
                    errors = max(1.0 / number, errors)
                    test_errors.append(errors)

            return model, sess.run(
                model.category_out,
                feed_dict={
                    model.emb_in: embeddings.getEmbeddings(),
                    model.dropout: 1.0
                }
            ), np.array(zip(fractions, test_errors))


def predictiveMultiClassWeights(examples, embeddings):
    has_unknown = True if len(examples) < 3 else False

    subset_A, subset_B = _splitTrainingAndTest(examples)
    model, weights_A, error_A = _doTraining(examples, subset_A, subset_B, embeddings, has_unknown)
    model, weights_B, error_B = _doTraining(examples, subset_B, subset_A, embeddings, has_unknown)
    weights = (weights_A + weights_B) / 2.
    max_error_length = min(error_A.shape[0], error_B.shape[0])
    error = (error_A[:max_error_length, :] + error_B[:max_error_length, :]) / 2
    print "predictiveMultiClassWeights weights.shape",weights.shape
    return weights, error
