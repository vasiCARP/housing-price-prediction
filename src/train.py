import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Citire date
df = pd.read_csv("data/train.csv")

# 2. Selectam cateva coloane simple
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Normalizare
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Model
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

# 6. Compilare
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# 7. Antrenare
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)

# 8. Evaluare
loss, mae = model.evaluate(X_test, y_test)
print("MAE:", mae)

# 9. Predictie
sample = np.array([[120, 3, 2]])
sample = scaler.transform(sample)

prediction = model.predict(sample)
print("Pret estimat:", prediction[0][0])
