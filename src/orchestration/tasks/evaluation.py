from prefect import task
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

@task
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    report = classification_report(y_val, y_pred)
    
    print(report)
    
    return acc, auc