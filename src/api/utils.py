import pickle

def load_model(model_path="src/models/log_reg.bin"): 
    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    return (dv, model)

def predict(model_bundle, input_dict): 
    dv, model = model_bundle
    X = dv.transform([input_dict])
    pred = model.predict(X)[0]
    return