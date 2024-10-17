import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
)
import matplotlib.pyplot as plt

from models.perceptrons.perceptron_non_linear import PerceptronNonLinear
from models.perceptrons.perceptron_linear import PerceptronLinear

# Cargar datos
data = pd.read_csv("../res/TP3-ej2-conjunto.csv")
X = data[["x1", "x2", "x3"]].values
y = data["y"].values

# Escalar los datos
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

seed = 42
num_features = X_scaled.shape[1]
learning_rate = 0.001
epsilon = 1e-5
non_linear_fn = "tanh"
beta = 2.0
num_epochs = 1000

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    # Initialize and train perceptrons
    perceptron = PerceptronNonLinear(
        seed=seed,
        num_features=num_features,
        learning_rate=learning_rate,
        epsilon=epsilon,
        non_linear_fn=non_linear_fn,
        beta=beta,
    )

    perceptron_linear = PerceptronLinear(
        seed=seed,
        num_features=num_features,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )

    perceptron.fit(X_train, y_train, num_epochs)
    perceptron_linear.fit(X_train, y_train, num_epochs)

    # Make predictions
    y_pred_scaled = perceptron.predict(X_test)
    y_pred_scaled_linear = perceptron_linear.predict(X_test)

    # Inverse transform predictions and targets to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_linear = scaler_y.inverse_transform(
        y_pred_scaled_linear.reshape(-1, 1)
    ).ravel()

    # Evaluate model using regression metrics
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
    med_ae = median_absolute_error(y_test_original, y_pred)

    mse_linear = mean_squared_error(y_test_original, y_pred_linear)
    mae_linear = mean_absolute_error(y_test_original, y_pred_linear)
    r2_linear = r2_score(y_test_original, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)
    mape_linear = (
        np.mean(np.abs((y_test_original - y_pred_linear) / y_test_original)) * 100
    )
    med_ae_linear = median_absolute_error(y_test_original, y_pred_linear)

    results.append(
        {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "mape": mape,
            "med_ae": med_ae,
            "mse_linear": mse_linear,
            "mae_linear": mae_linear,
            "r2_linear": r2_linear,
            "rmse_linear": rmse_linear,
            "mape_linear": mape_linear,
            "med_ae_linear": med_ae_linear,
        }
    )

avg_mse = np.mean([res["mse"] for res in results])
avg_mae = np.mean([res["mae"] for res in results])
avg_r2 = np.mean([res["r2"] for res in results])
avg_rmse = np.mean([res["rmse"] for res in results])
avg_mape = np.mean([res["mape"] for res in results])
avg_med_ae = np.mean([res["med_ae"] for res in results])

avg_mse_linear = np.mean([res["mse_linear"] for res in results])
avg_mae_linear = np.mean([res["mae_linear"] for res in results])
avg_r2_linear = np.mean([res["r2_linear"] for res in results])
avg_rmse_linear = np.mean([res["rmse_linear"] for res in results])
avg_mape_linear = np.mean([res["mape_linear"] for res in results])
avg_med_ae_linear = np.mean([res["med_ae_linear"] for res in results])

print("Average metrics for non-linear perceptron:")
print(f"MSE: {avg_mse}")
print(f"MAE: {avg_mae}")
print(f"R2: {avg_r2}")
print(f"RMSE: {avg_rmse}")
print(f"MAPE: {avg_mape}")
print(f"Median AE: {avg_med_ae}")

print("Average metrics for linear perceptron:")
print(f"MSE: {avg_mse_linear}")
print(f"MAE: {avg_mae_linear}")
print(f"R2: {avg_r2_linear}")
print(f"RMSE: {avg_rmse_linear}")
print(f"MAPE: {avg_mape_linear}")
print(f"Median AE: {avg_med_ae_linear}")

metrics = ["MSE", "MAE", "R2", "RMSE", "MAPE", "Median AE"]
non_linear_values = [avg_mse, avg_mae, avg_r2, avg_rmse, avg_mape, avg_med_ae]
linear_values = [
    avg_mse_linear,
    avg_mae_linear,
    avg_r2_linear,
    avg_rmse_linear,
    avg_mape_linear,
    avg_med_ae_linear,
]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width / 2, non_linear_values, width, label="Non-Linear Perceptron")
rects2 = ax.bar(x + width / 2, linear_values, width, label="Linear Perceptron")

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel("Metrics")
titel_string = (
    f"Performance Comparison: Non-Linear vs Linear Perceptron ({num_epochs} epochs)\n"
    + f"$\\eta =${learning_rate} - Non-Linear: {non_linear_fn} activation - $\\beta =$ {beta} - seed: {seed} \n"
)
ax.set_title(titel_string)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(round(height, 2)),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  #
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

plt.tight_layout()
plt.show()

max_values = [max(non_linear_values[i], linear_values[i]) for i in range(len(metrics))]
non_linear_normalized = [
    non_linear_values[i] / max_values[i] for i in range(len(metrics))
]
linear_normalized = [linear_values[i] / max_values[i] for i in range(len(metrics))]

fig, ax = plt.subplots(figsize=(12, 6))

rects1 = ax.bar(
    x - width / 2, non_linear_normalized, width, label="Non-Linear Perceptron"
)
rects2 = ax.bar(x + width / 2, linear_normalized, width, label="Linear Perceptron")

ax.set_xlabel("Metrics")
titel2_string = (
    f"Performance Comparison: Non-Linear vs Linear Perceptron ({num_epochs} epochs) - Normalized\n"
    + f"$\\eta =${learning_rate} - Non-Linear: {non_linear_fn} activation - $\\beta =$ {beta} - seed: {seed} \n"
)
ax.set_title(titel2_string)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(round(height, 2)),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

# Display the second graph
plt.tight_layout()
plt.show()
