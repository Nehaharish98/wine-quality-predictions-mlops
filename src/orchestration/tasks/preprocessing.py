from prefect import task
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

@task
def preprocess_data(df):
    df["good_quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

    numerical = df.columns.drop(['quality', 'good_quality']).tolist()
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["good_quality"]
    )
    
    train_dicts = train_df[numerical].to_dict(orient='records')
    val_dicts = val_df[numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = train_df["good_quality"].values
    y_val = val_df["good_quality"].values

    return X_train, X_val, y_train, y_val, dv
