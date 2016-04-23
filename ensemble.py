"""For training and predicting via final ensemble."""

import common
import extract_features

import numpy as np
import random
import xgboost as xgb

TRAIN_FOLDER = './data/train'
VALIDATE_FOLDER = './data/validate'
LABELS_FILE = './data/train.csv'
MODEL_PREFIX = './xgb/model'
PREDICT_CDF = False  # whether to predict 600-val CDF or actual label
SEED = 888
TEST_PROP = 0.15
NUM_ESTIMATORS = 100

# see http://xgboost.readthedocs.org/en/latest/parameter.html
PARAMETERS = {
    'bst:max_depth': 3,
    'bst:eta': 0.04,
    'nthread': 8,
    }


def train(data_folder=TRAIN_FOLDER,
          labels_file=LABELS_FILE,
          model_prefix=MODEL_PREFIX,
          cdf_prediction=PREDICT_CDF,
          parameters=PARAMETERS):
    """Train ensemble."""

    random.seed(SEED)
    _, sys_lab, dia_lab, features = \
        extract_features.extract_features(data_folder, labels_file)
    assert len(sys_lab) == len(dia_lab) == features.shape[0]

    print 'total data', features.shape[0]
    print 'num features', features.shape[1]

    random.seed(SEED)
    test_prop = TEST_PROP
    test_size = int(features.shape[0] * test_prop)
    shuffled_indices = random.shuffle(range(features.shape[0]))

    test_sys_lab = sys_lab[:test_size]
    test_dia_lab = dia_lab[:test_size]
    test_features = features[:test_size]

    train_sys_lab = sys_lab[test_size:]
    train_dia_lab = dia_lab[test_size:]
    train_features = features[test_size:]

    print 'train size', train_features.shape[0]
    print 'test size', test_features.shape[0]

    systole_prefix = '%s_sys' % model_prefix
    diastole_prefix = '%s_dia' % model_prefix

    print 'training systole model'
    _train(train_features, train_sys_lab, test_features, test_sys_lab,
           model_prefix=systole_prefix,
           cdf_prediction=cdf_prediction,
           parameters=parameters)

    print 'training diastole model'
    _train(train_features, train_dia_lab, test_features, test_dia_lab,
           model_prefix=diastole_prefix,
           cdf_prediction=cdf_prediction,
           parameters=parameters)

    print 'final systole evaluation on test set'
    sys_crps = evaluate(test_features, test_sys_lab,
                        model_prefix=systole_prefix,
                        cdf_prediction=cdf_prediction)

    print 'final diastole evaluation on test set'
    dia_crps = evaluate(test_features, test_dia_lab,
                        model_prefix=diastole_prefix,
                        cdf_prediction=cdf_prediction)

    print 'test sys crps', sys_crps
    print 'test dia crps', dia_crps
    print 'overall test crps', (sys_crps + dia_crps) * 0.5


def _train(train_features, train_lab, test_features, test_lab,
           model_prefix=MODEL_PREFIX,
           cdf_prediction=PREDICT_CDF,
           parameters=PARAMETERS):

    if not cdf_prediction:
        params = dict(parameters)
        params['objective'] = 'reg:linear'

        dtrain = xgb.DMatrix(train_features, label=train_lab)
        dtest = xgb.DMatrix(test_features, label=test_lab)

        evallist  = [(dtrain, 'train'), (dtest, 'test')]

        def feval(pred, data):
            labels = data.get_label()
            return 'crps', np.mean(
                [common.crps_sample_sample(l, p)
                 for l, p in zip(labels, pred)])

        model = xgb.train(params, dtrain,
                          NUM_ESTIMATORS,
                          evallist,
                          feval=feval)

        file_name = '%s.model' % model_prefix
        print 'saving model to', file_name
        model.save_model(file_name)

    else:
        for i in xrange(600):
            print 'training CDF at i=%d' % i
            params = dict(parameters)
            params['objective'] = 'binary:logistic'

            dtrain = xgb.DMatrix(train_features, label=(train_lab <= i))
            dtest = xgb.DMatrix(test_features, label=(test_lab <= i))

            evallist  = [(dtrain, 'train'), (dtest, 'test')]

            model = xgb.train(params, dtrain,
                              NUM_ESTIMATORS,
                              evallist)

            file_name = '%s_cdf%d.model' % (model_prefix, i)
            print 'saving model to', file_name
            model.save_model(file_name)


def load_model(model_prefix, cdf_prediction):
    """Loads model(s)."""
    if not cdf_prediction:
        model = xgb.Booster({'nthread': 4})  # init model
        model.load_model('%s.model' % model_prefix)
        return model

    else:
        models = []
        for i in xrange(600):
            model = xgb.Booster({'nthread': 4})  # init model
            model.load_model('%s_cdf%d.model' % (model_prefix, i))
            models.append(model)
        return models


def predict_cdfs(features, model_prefix, cdf_prediction):
    """Get CDFs predicted by model."""
    ndata = features.shape[0]
    features = xgb.DMatrix(features)
    if not cdf_prediction:
        model = load_model(model_prefix, cdf_prediction)
        pred_label = model.predict(features)
        return np.array([
            (pred_label[i] <= np.arange(600))
            for i in xrange(len(pred_label))])

    else:
        models = load_model(model_prefix, cdf_prediction)
        pred_labels = [models[i].predict(features)
                       for i in xrange(600)]

        # ensure CDF criteria
        cdfs = [[] for _ in xrange(ndata)]
        for i in xrange(600):
            pred = pred_labels[i]
            for k, cdf in enumerate(cdfs):
                if i == 0:
                    cdf.append(max(min(pred[k], 1.0), 0.0))
                else:
                    cdf.append(max(min(pred[k], 1.0), cdf[-1]))

        return np.array(cdfs)


def evaluate(features, labels, model_prefix, cdf_prediction):
    """Evaluate model and return CRPS."""
    cdfs = predict_cdfs(features, model_prefix, cdf_prediction)
    total_crps = 0.0
    for i in xrange(cdfs.shape[0]):
        total_crps += common.crps_sample(labels[i], cdfs[i])

    crps = total_crps / len(labels)
    return crps


def validate(data_folder=VALIDATE_FOLDER,
             model_prefix=MODEL_PREFIX,
             cdf_prediction=PREDICT_CDF):
    """Use ensemble to predict CDFs on validation set."""

    study_ids, sys_lab, dia_lab, features = \
        extract_features.extract_features(data_folder)
    assert study_ids == tuple(range(501, 701))
    assert len(sys_lab) == len(dia_lab) == features.shape[0]

    print 'total data', features.shape[0]
    print 'num features', features.shape[1]

    systole_prefix = '%s_sys' % model_prefix
    diastole_prefix = '%s_dia' % model_prefix

    systole_cdfs = predict_cdfs(features, systole_prefix, cdf_prediction)
    diastole_cdfs = predict_cdfs(features, diastole_prefix, cdf_prediction)
    return systole_cdfs, diastole_cdfs
