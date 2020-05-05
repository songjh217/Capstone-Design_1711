결과 데이터 엑셀에 정리한 것들


df1 = pd.DataFrame([[RMSE_1, RMSE_2, RMSE_3, RMSE_4],[MAPE_1,MAPE_2,MAPE_3,MAPE_4]])
df2 = pd.DataFrame(prediction ...)
df.to_excel('data2.xlsx', index=False) 

