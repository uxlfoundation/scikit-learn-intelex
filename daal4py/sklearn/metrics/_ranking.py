import daal4py as d4p
import numpy as np
from sklearn.utils import column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import label_binarize
from sklearn.metrics._ranking import _multiclass_roc_auc_score, _binary_roc_auc_score
from sklearn.metrics._base import _average_binary_score, _average_multiclass_ovo_score
from .._utils import get_patch_message
import logging
from functools import partial

def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None):
    y_type = type_of_target(y_true)
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    if y_type == "multiclass" or (y_type == "binary" and
                                  y_score.ndim == 2 and
                                  y_score.shape[1] > 2):
        # do not support partial ROC computation for multiclass
        if max_fpr is not None and max_fpr != 1.:
            raise ValueError("Partial AUC computation not available in "
                             "multiclass setting, 'max_fpr' must be"
                             " set to `None`, received `max_fpr={0}` "
                             "instead".format(max_fpr))
        if multi_class == 'raise':
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        return _multiclass_roc_auc_score(y_true, y_score, labels,
                                         multi_class, average, sample_weight)
    elif y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, classes=labels)[:, 0]
        if len(labels) != 2:
            raise ValueError("Only one class present in y_true. ROC AUC score "
                         "is not defined in that case.")
        if max_fpr is None and sample_weight is None:
            logging.info("sklearn.metrics.roc_auc_score.binary: " + get_patch_message("daal"))
            if y_score.dtype == np.float64:
                return d4p.daal_roc_auc_score(y_true.reshape(1, -1), y_score.reshape(1, -1), 1)
            else:
                return d4p.daal_roc_auc_score(y_true.reshape(1, -1), y_score.reshape(1, -1), 0)
        logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("sklearn"))
        return _average_binary_score(partial(_binary_roc_auc_score,
                                             max_fpr=max_fpr),
                                     y_true, y_score, average,
                                     sample_weight=sample_weight)
    else:  # multilabel-indicator
        return _average_binary_score(partial(_binary_roc_auc_score,
                                             max_fpr=max_fpr),
                                     y_true, y_score, average,
                                     sample_weight=sample_weight)
    #return d4p.daal_roc_auc_score(y_true.reshape(1, -1), y_score.reshape(1, -1), 1)