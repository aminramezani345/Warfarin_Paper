# Calculate Mean Squared Error (MSE)
mse_DTree = mean_squared_error(y_test, y_pred_DTree)
print("Mean Squared Error (MSE_DTree):", mse_DTree)

# Calculate Mean Absolute Error (MAE)
mae_DTree = mean_absolute_error(y_test, y_pred_DTree)
print("Mean Absolute Error (MAE_DTree):", mae_DTree)

def mean_absolute_percentage_error(y_true, y_pred_DTree):
    return np.mean(np.abs((y_true - y_pred_DTree) / y_true)) * 100

# Calculate Mean Absolute Percentage Error (MAPE)
mape_DTree = mean_absolute_percentage_error(y_test, y_pred_DTree)
print("Mean Absolute Percentage Error (MAPE_DTree):", mape_DTree)

# Calculate R-squared
r_squared_DTree = r2_score(y_test, y_pred_DTree)
print("R-squared:", r_squared_DTree)

# Calculate the correlation between y_true and y_pred
correlation_DTree = np.corrcoef(y_test, y_pred_DTree)[0, 1]
print("Correlation between y_true and y_pred_DTree:", correlation_DTree)

############################################################
# Summary of the results from different models
results = {
    "Model": ["SVR", "KNN", "Linear Regression", "Decision Tree"],
    "MSE": [mean_squared_error(y_test, y_pred_svr), mse_knn, mse_Lreg, mse_DTree],
    "MAE": [mean_absolute_error(y_test, y_pred_svr), mae_knn, mae_Lreg, mae_DTree],
    "MAPE": [mape_svr, mape_knn, mape_Lreg, mape_DTree],
    "R-squared": [score_svr, r_squared_knn, r_squared_Lreg, r_squared_DTree],
    "Correlation": [correlation_svr, correlation_knn, correlation_Lreg, correlation_DTree]
}

results_df = pd.DataFrame(results)
print(results_df)
