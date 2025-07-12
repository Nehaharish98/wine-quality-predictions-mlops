from prefect import task
from sklearn.linear_model import LogisticRegression
import pickle

@task
def train_logistic_regression(X_train, y_train, X_val, y_val, dv):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    with open("models/log_reg.bin", "wb") as f_out:
        pickle.dump((dv, model), f_out)

    return model
