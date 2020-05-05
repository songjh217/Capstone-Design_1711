df1 = pd.DataFrame([[RMSE_1, RMSE_2, RMSE_3, RMSE_4],[MAPE_1,MAPE_2,MAPE_3,MAPE_4]]) 
df2 = pd.DataFrame(prediction_powerV_nn) 
df1.to_excel('maxdata1.xlsx', index=False)
df2.to_excel('maxdata2.xlsx', index=False)
