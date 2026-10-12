"""Per-customer explanations: additive log-odds contribution of each feature.
Logistic regression -> exact linear terms. XGBoost -> built-in SHAP (pred_contribs). Other trees -> shap.TreeExplainer."""
import numpy as np

def contributions(name, model, X):
    if name == "logistic_regression":
        scaler, lr = model.steps[0][1], model.steps[-1][1]
        return scaler.transform(X) * lr.coef_[0]
    if name == "xgboost":
        import xgboost as xgb
        c = model.get_booster().predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
        return np.asarray(c)[:, :-1]
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv
