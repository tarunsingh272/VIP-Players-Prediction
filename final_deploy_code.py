import os
import math as mh
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pypyodbc #Connecting to odbc source
import pandas.io.sql as psql #Connecting pandas directly to DB
from datetime import datetime as DateTime, timedelta as TimeDelta
import pickle

# For visualizations
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic("config InlineBackend.figure_format = 'png'")
get_ipython().magic('matplotlib inline')

# For data parsing
from datetime import datetime

# For choosing attributes that have good gaussian distribution
from scipy.stats import shapiro

# Needed for getting parameters for models
from sklearn.cross_validation import LeaveOneOut
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

# Models
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier

# For scaling/normalizing values
from sklearn.preprocessing import MinMaxScaler

#For splitting data in train and test set; 
from sklearn.model_selection import train_test_split


connection = pypyodbc.connect('Driver={SQL Server};'
                                'Server=X.X.X.X;'
                                'Database=Dump_Tables;'
                                'uid=X.X.X.X;pwd=X.X.X.X')


#today_date = datetime.today().strftime('%Y-%m-%d')
#today_date = '2017-12-25'

date_1 = DateTime.today()
end_date = date_1 + TimeDelta(days=-1)
today_date = end_date.strftime('%Y-%m-%d')
print(today_date)


r2_query = '''select users.userID,users.firstDepositDate,sum(case when r2.[date] between users."7thDay" and users."13thDay" then CashGameCount end) as '7to14_CashGames',count(distinct case when r2.[date] between users."7thDay" and users."13thDay" then CONVERT(DATE, r2.[date]) end) as '7to14_ActiveDays',Sum(case when r2.[date] between users."7thDay" and users."13thDay" then GameWon end) as '7to14_WinCashGames',
Sum(case when r2.[date]between users."7thDay" and users."13thDay" then GameLost end) as '7to14_LossCashGames',avg(case when r2.[date] between users."7thDay" and users."13thDay" then r2.entryFee end) as '7to14_AvgEntryFee',avg(case when r2.[date] between users."7thDay" and users."13thDay" then r2.seat end) as '7to14_AvgComposition',Sum(case when r2.[date] between users."7thDay" and users."13thDay" then Rake end) as '7to14_RakeGenerated',sum(case when r2.[date] between users."14thDay" and users."21stDay" then CashGameCount end) as '14to21_CashGames',count(distinct case when r2.[date] between users."14thDay" and users."21stDay" then CONVERT(DATE, r2.[date]) end) as '14to21_ActiveDays',SUM(case when r2.[date] between users."14thDay" and users."21stDay"  then r2.GameWon end) as '14to21_WinCashGames',SUM(case when r2.[date] between users."14thDay" and users."21stDay" then r2.GameLost end) as '14to21_LossCashGames',avg(case when r2.[date] between users."14thDay" and users."21stDay" then r2.EntryFee end) as '14to21_AvgEntryFee',avg(case when r2.[date] between users."14thDay" and users."21stDay" then r2.seat end) as '14to21_AvgComposition',SUM(case when r2.[date] between users."14thDay" and users."21stDay"  then r2.Rake end) as '14to21_RakeGenerated' from Dump_Tables.dbo.R2New r2 join (select userID,CONVERT(DATE, firstDepositDate) as firstDepositDate,Dateadd(dd,7,CONVERT(DATE, firstDepositDate)) as '7thDay',dateadd(dd,13,CONVERT(DATE, firstDepositDate)) as '13thDay',Dateadd(dd,14,CONVERT(DATE, firstDepositDate)) as '14thDay',Dateadd(dd,21,CONVERT(DATE, firstDepositDate)) as '21stDay' from JWR.dbo.Users where firstDepositDate is not NULL and userID != -1 and CONVERT(DATE, firstDepositDate) = dateadd(dd,-21,CONVERT(DATE,\''''+today_date+'''\'))) users on users.userID=r2.userId and r2.[date] between users."7thDay" and users."21stDay" group by users.userID,users.firstDepositDate;'''


deposit_data_query = '''select dep.userID,users.firstDepositAmount as 'First_Deposit',isnull(sum(case when CONVERT(DATE,dep.txnCreditedTime) between users."7thDay" and users."13thDay" then dep.amount end),0) as '7to14_DepositsAmount',count(case when CONVERT(DATE,dep.txnCreditedTime) between users."7thDay" and users."13thDay" then dep.internalTransactionID end) as '7to14_DepositsCount',isnull(sum(case when CONVERT(DATE,dep.txnCreditedTime) between users."14thDay" and users."21stDay" then amount end),0) as '14to21_DepositsAmount',count(case when CONVERT(DATE,dep.txnCreditedTime) between users."14thDay" and users."21stDay" then dep.internalTransactionID end) as '14to21_DepositsCount' from JWR.dbo.UserDeposits dep join (select userID,firstDepositAmount,CONVERT(DATE, firstDepositDate) as firstGamedt,Dateadd(dd,7,CONVERT(DATE, firstDepositDate)) as '7thDay',Dateadd(dd,13,CONVERT(DATE, firstDepositDate)) as '13thDay',Dateadd(dd,14,CONVERT(DATE, firstDepositDate)) as '14thDay',Dateadd(dd,21,CONVERT(DATE, firstDepositDate)) as '21stDay' from JWR.dbo.Users where firstDepositDate is not NULL and userID != -1 and CONVERT(DATE, firstDepositDate) = dateadd(dd,-21,CONVERT(DATE,\''''+today_date+'''\'))) users on users.userID=dep.userId and dep.txnCreditedTime between users."7thDay" and users."21stDay" group by dep.userID,users.firstDepositAmount'''


withdraw_data_query = '''select wdh.userID,isnull(sum(case when CONVERT(DATE,wdh."timeStampFulfilled") between users."7thDay" and users."13thDay" then wdh.amount end),0) as '7to14_WdhAmount',count(case when CONVERT(DATE,wdh."timeStampFulfilled") between users."7thDay" and users."13thDay" then wdh.WithdrawalID end) as '7to14_WdhCount',isnull(sum(case when CONVERT(DATE,wdh."timeStampFulfilled") between users."14thDay" and users."21stDay" then wdh.amount end),0) as '14to21_WdhAmount', count(case when CONVERT(DATE,wdh."timeStampFulfilled") between users."14thDay" and users."21stDay" then wdh.WithdrawalID end) as '14to21_WdhCount' from JWR.dbo.Withdrawals wdh join ( select userID,CONVERT(DATE, firstDepositDate) as firstGamedt,Dateadd(dd,7,CONVERT(DATE,firstDepositDate)) as '7thDay', Dateadd(dd,13,CONVERT(DATE, firstDepositDate)) as '13thDay',Dateadd(dd,14,CONVERT(DATE, firstDepositDate)) as '14thDay',Dateadd(dd,21,CONVERT(DATE, firstDepositDate)) as '21stDay' from JWR.dbo.Users where firstDepositDate is not NULL and userID != -1 and CONVERT(DATE, firstDepositDate) = dateadd(dd,-21,CONVERT(DATE,\''''+today_date+'''\'))) users on users.userID=wdh.userId and wdh."timeStampFulfilled" between users."7thDay" and users."21stDay" where timestampFulfilled is not null group by wdh.userID'''


r2 = psql.read_sql(r2_query, connection)
deposit_data = psql.read_sql(deposit_data_query, connection)
withdraw_data = psql.read_sql(withdraw_data_query, connection)


#deposit_data
connection.close()


final_df_test = pd.merge(r2,deposit_data,on='userid',how='left') #Merge deposit data
final_df_test = pd.merge(final_df_test,withdraw_data,on='userid',how='left') #Merge withdrawal data


#Replace all NaN with 0
final_df_test=final_df_test.fillna(0)

final_df_test.shape

X_test2 = final_df_test.drop(['firstdepositdate','userid'],axis=1)


loaded_model = pickle.load(open("pima.pickle.dat", "rb"))

y_pred2 = loaded_model.predict(X_test2)


predicted2 = pd.DataFrame(columns=['Prediction'],index=final_df_test.index, data=y_pred2)

new_predictions = pd.concat([final_df_test, predicted2], axis=1)

#new_predictions


connection_write = pypyodbc.connect('Driver={SQL Server};'
                                'Server=X.X.X.X;'
                                'Database=Dump_Tables;'
                                'uid=X.X.X.X;pwd=X.X.X.X')

cursor2 = connection_write.cursor()
SQLCommand = ("INSERT INTO Dump_Tables.dbo.vip_prediction" "(userid, firstdepositdate, prediction_category) " "VALUES (?,?,?)")
row_iterator = new_predictions.iterrows()
for i, row in row_iterator:
    Values = [row['userid'],row['firstdepositdate'],row['Prediction']]
    cursor2.execute(SQLCommand,Values)
    connection_write.commit()

connection_write.close()