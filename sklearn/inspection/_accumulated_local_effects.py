"""accumulated local effect plots for regression and classification models."""

# Authors: Cameron Lyons
# License: BSD 3 clause

import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats.mstats import mquantiles

from ..base import is_classifier, is_regressor
from ..pipeline import Pipeline
from ..utils.extmath import cartesian
from ..utils import check_array
from ..utils import check_matplotlib_support  # noqa
from ..utils import _safe_indexing
from ..utils import _determine_key_type
from ..utils import _get_column_indices
from ..utils.validation import check_is_fitted
from ..utils import Bunch



__all__ = [
    'accumulated_local_effects',
]

def _grid_from_X(X, grid_resolution):
    """Generate a grid of points based on the percentiles of X.

    The grid is a cartesian product between the columns of ``values``. The
    ith column of ``values`` consists in ``grid_resolution`` quantiles
    between  of the jth column of X.
    If ``grid_resolution`` is bigger than the number of unique values in the
    jth column of X, then those unique values will be used instead.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_target_features)
        The data

    grid_resolution : int
        The number of equally spaced points to be placed on the grid for each
        feature.

    Returns
    -------
    grid : ndarray, shape (n_points, n_target_features)
        A value for each feature at each point in the grid. ``n_points`` is
        always ``<= grid_resolution ** X.shape[1]``.

    values : list of 1d ndarrays
        The values with which the grid has been created. The size of each
        array ``values[j]`` is either ``grid_resolution``, or the number of
        unique values in ``X[:, j]``, whichever is smaller.
    """

    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")

    values = []
    for feature in range(X.shape[1]):
        uniques = np.unique(_safe_indexing(X, feature, axis=1))
        if uniques.shape[0] < grid_resolution:
            # feature has low resolution use unique vals
            emp_percentiles = uniques
        else:
            # create axis based on percentiles and grid resolution
            emp_percentiles = mquantiles(
                _safe_indexing(X, feature, axis=1), prob=np.linspace(0, 1, grid_resolution), axis=0
            )

        values.append(emp_percentiles)

    return cartesian(values), values

def _calculate_accumulated_local_effects(est, grid, features, X, response_method):
    predictions = []
    averaged_predictions = []

    # define the prediction_method (predict, predict_proba, decision_function).
    if is_regressor(est):
        prediction_method = est.predict
    else:
        predict_proba = getattr(est, 'predict_proba', None)
        decision_function = getattr(est, 'decision_function', None)
        if response_method == 'auto':
            # try predict_proba, then decision_function if it doesn't exist
            prediction_method = predict_proba or decision_function
        else:
            prediction_method = (predict_proba if response_method ==
                                 'predict_proba' else decision_function)
        if prediction_method is None:
            if response_method == 'auto':
                raise ValueError(
                    'The estimator has no predict_proba and no '
                    'decision_function method.'
                )
            elif response_method == 'predict_proba':
                raise ValueError('The estimator has no predict_proba method.')
            else:
                raise ValueError(
                    'The estimator has no decision_function method.')
    X_eval = X.copy()
    a1 = pd.cut(X_eval.iloc[:, 0], bins=grid[0], labels=False).fillna(0.0).astype(int)
    X_eval_2 = X.copy()
    X_eval_2.iloc[:, 0] = grid[0][a1 + 1]
    y_hat = prediction_method(X_eval)
    y_hat_2 = prediction_method(X_eval_2)
    delta = y_hat_2 - y_hat
    return averaged_predictions, predictions

def accumulated_local_effects(estimator, X, features, *,
                              response_method='auto',
                              grid_resolution=100,
                              kind='legacy'):

    """Accumulated Local Effects of ``features``.

    Accumulated Local Effects of a feature (or a set of features) corresponds to
    the average response of an estimator for each possible value of the
    feature.


    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    X : {array-like or dataframe} of shape (n_samples, n_features)
        ``X`` is used to generate a grid of values for the target
        ``features`` (where the partial dependence will be evaluated), and
        also to generate values for the complement features when the
        `method` is 'brute'.

    features : array-like of {int, str}
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

    response_method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is 'recursion', the response is always the output of
        :term:`decision_function`.

    grid_resolution : int, default=100
        The number of equally spaced points on the grid, for each target
        feature.

    kind : {'legacy', 'average', 'individual', 'both'}, default='legacy'
        Whether to return the partial dependence averaged across all the
        samples in the dataset or one line per sample or both.
        See Returns below.

        Note that the fast `method='recursion'` option is only available for
        `kind='average'`. Plotting individual dependencies requires using the
        slower `method='brute'` option.

        .. versionadded:: 0.24
        .. deprecated:: 0.24
            `kind='legacy'` is deprecated and will be removed in version 0.26.
            `kind='average'` will be the new default. It is intended to migrate
            from the ndarray output to :class:`~sklearn.utils.Bunch` output.


    Returns
    -------
    predictions : ndarray or :class:`~sklearn.utils.Bunch`

        - if `kind='legacy'`, return value is ndarray of shape (n_outputs, \
                len(values[0]), len(values[1]), ...)
            The predictions for all the points in the grid, averaged
            over all samples in X (or over the training data if ``method``
            is 'recursion').

        - if `kind='individual'`, `'average'` or `'both'`, return value is \
                :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            individual : ndarray of shape (n_outputs, n_instances, \
                    len(values[0]), len(values[1]), ...)
                The predictions for all the points in the grid for all
                samples in X. This is also known as Individual
                Conditional Expectation (ICE)

            average : ndarray of shape (n_outputs, len(values[0]), \
                    len(values[1]), ...)
                The predictions for all the points in the grid, averaged
                over all samples in X (or over the training data if
                ``method`` is 'recursion').
                Only available when kind='both'.

            values : seq of 1d ndarrays
                The values with which the grid has been created. The generated
                grid is a cartesian product of the arrays in ``values``.
                ``len(values) == len(features)``. The size of each array
                ``values[j]`` is either ``grid_resolution``, or the number of
                unique values in ``X[:, j]``, whichever is smaller.

        ``n_outputs`` corresponds to the number of classes in a multi-class
        setting, or to the number of tasks for multi-output regression.
        For classical regression and binary classification ``n_outputs==1``.
        ``n_values_feature_j`` corresponds to the size ``values[j]``.

    values : seq of 1d ndarrays
        The values with which the grid has been created. The generated grid
        is a cartesian product of the arrays in ``values``. ``len(values) ==
        len(features)``. The size of each array ``values[j]`` is either
        ``grid_resolution``, or the number of unique values in ``X[:, j]``,
        whichever is smaller. Only available when `kind="legacy"`.

    Examples
    --------
    >>> X = [[0, 0, 2], [1, 0, 0]]
    >>> y = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    >>> accumulated_local_effects(gb, features=[0], X=X, percentiles=(0, 1),
    ...                    grid_resolution=2) # doctest: +SKIP
    (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])

    See also
    --------
    sklearn.inspection.plot_accumulated_local_effects: Plot accumulated local effeccts
    """

    if not (is_classifier(estimator) or is_regressor(estimator)):
        raise ValueError(
            "'estimator' must be a fitted regressor or classifier."
        )

    if isinstance(estimator, Pipeline):
        # TODO: to be removed if/when pipeline get a `steps_` attributes
        # assuming Pipeline is the only estimator that does not store a new
        # attribute
        for est in estimator:
            # FIXME: remove the None option when it will be deprecated
            if est not in (None, 'drop'):
                check_is_fitted(est)
    else:
        check_is_fitted(estimator)

    if (is_classifier(estimator) and
            isinstance(estimator.classes_[0], np.ndarray)):
        raise ValueError(
            'Multiclass-multioutput estimators are not supported'
        )

    # Use check_array only on lists and other non-array-likes / sparse. Do not
    # convert DataFrame into a NumPy array.
    if not(hasattr(X, '__array__') or sparse.issparse(X)):
        X = check_array(X, force_all_finite='allow-nan', dtype=object)

    accepted_responses = ('auto', 'predict_proba', 'decision_function')
    if response_method not in accepted_responses:
        raise ValueError(
            'response_method {} is invalid. Accepted response_method names '
            'are {}.'.format(response_method, ', '.join(accepted_responses)))

    if is_regressor(estimator) and response_method != 'auto':
        raise ValueError(
            "The response_method parameter is ignored for regressors and "
            "must be 'auto'."
        )

    if _determine_key_type(features, accept_slice=False) == 'int':
        # _get_column_indices() supports negative indexing. Here, we limit
        # the indexing to be positive. The upper bound will be checked
        # by _get_column_indices()
        if np.any(np.less(features, 0)):
            raise ValueError(
                'all features must be in [0, {}]'.format(X.shape[1] - 1)
            )

    features_indices = np.asarray(
        _get_column_indices(X, features), dtype=np.int32, order='C'
    ).ravel()

    grid, values = _grid_from_X(
        _safe_indexing(X, features_indices, axis=1),
        grid_resolution
    )

    averaged_predictions, predictions = _calculate_accumulated_local_effects(
            estimator, grid, features_indices, X, response_method
        )

        # reshape predictions to
        # (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
    #predictions = predictions.reshape(
    #        -1, X.shape[0], *[val.shape[0] for val in values]
    #    )

    if kind == 'legacy':
        warnings.warn(
            "A Bunch will be returned in place of 'predictions' from version"
            " 0.26 with partial dependence results accessible via the "
            "'average' key. In the meantime, pass kind='average' to get the "
            "future behaviour.",
            FutureWarning
        )
        # TODO 0.26: Remove kind == 'legacy' section
        return averaged_predictions, values
    elif kind == 'average':
        return Bunch(average=averaged_predictions, values=values)
    elif kind == 'individual':
        return Bunch(individual=predictions, values=values)
    else:  # kind='both'
        return Bunch(
                average=averaged_predictions, individual=predictions,
                values=values,
            )
