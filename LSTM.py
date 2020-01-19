import pandas as pd
import numpy as np
import math
import warnings
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import dash_core_components as dcc
import plotly.graph_objs as go
import matplotlib.pyplot as plt



from MySQL import *
np.random.seed(1234)
PYTHONHASHSEED = 0
warnings.filterwarnings('ignore')

work = dataSQL()

#n
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def Create_new_added(dataframe,code2,code3,code4,code5,interval): #Alarm type_code1, #feature 640 value
    #make additional copy of dataframe to extract features
    added_feature=dataframe.copy()
    added_featured=dataframe.copy()

#    dataframe["week_no"] = pd.to_datetime(dataframe["log_time"]).dt.week
    dataframe["log_time"] = pd.to_datetime(dataframe["log_time"])
    dataframe["Alarm"]=dataframe["Code"]==code2
    dataframe.loc[dataframe["Alarm"]==True,"Alarm"]=1
    dataframe.loc[dataframe["Alarm"]==False,"Alarm"]=0

    #resampling on weekly basis, the size should remain the same to realistically
    #show evolution over time
    data_f=dataframe.drop(columns=['A_W', 'Code', 'GpsPos', 'Value', 'LAT', 'LON'])
    d=(data_f.resample(interval,on="log_time").sum()).reset_index()
    #interval = 'D'
    time_=d['log_time']
    d["week_no"] = d.index

    d=d.drop(columns='log_time').copy()
    dataset_initial = d.values

    DataUV_week37_max=add_feature(added_feature,code3,interval)
    Data_on_of=add_feature_indicate_on(added_featured,code3,interval) #on or off?

    Data_discharge=add_feature_w(added_feature,code4,interval) #charge
    Data_charge=add_feature_w(added_feature,code5,interval)#discharge
    d=list(d['Alarm'])

    #prepare for recombination
    Combined=pd.DataFrame(data={'AlarmCount':d,'UVmin_week':DataUV_week37_max,
        'DischargeWater':Data_discharge,'Indicate':Data_on_of,
                                'log_time':time_})

    Combined=Combined.fillna(0)
    Combined_=Combined
    Combined=Combined.drop(columns=['log_time'])
    Combined=Combined.values
    dataset = Combined.astype('float32')
    return dataset,Combined_

def add_feature_w(added_feature,code,interval):
    #water  charge
    added_feature=added_feature[added_feature.Code==int(code)]
    added_feature["log_time"] = pd.to_datetime(added_feature["log_time"])
    added_feature=added_feature.resample(interval,on="log_time").sum().reset_index()
    added_feature["week_no"] = added_feature.index

    sum_weeks = list(added_feature['Value'])
    return sum_weeks

def add_feature(added_feature,code,interval):
    # maximum values of UV mean per day over the 7 day week period
    added_feature=added_feature[added_feature.Code==int(code)]
    added_feature["log_time"] = pd.to_datetime(added_feature["log_time"])
    added_feature=added_feature.resample(interval,on="log_time").mean().reset_index()
    added_feature["week_no"] = added_feature.index
    mean_weeks = list(added_feature['Value'])

    return mean_weeks


def add_feature_indicate_on(added_feature,code,interval):
    # maximum values of UV mean per day over the 7 day week period
    added_feature=added_feature[added_feature.Code==int(code)]
    added_feature["log_time"] = pd.to_datetime(added_feature["log_time"])
    added_feature=added_feature.resample(interval,on="log_time").mean().reset_index()
    added_feature["week_no"] = added_feature.index

    new=added_feature.copy()
    new["Value"] =new["Value"] >0

    new.loc[new["Value"]==True,"Value"]=1
    new.loc[new["Value"]==False,"Value"]=0

    mean_weeks =(list(new["Value"]))
    return mean_weeks


def scale(dataset):
    # split into train and test sets
    train_size = int(len(dataset) * 0.82)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    train = np.reshape(train, (train.shape[0], train.shape[1]))
    test = np.reshape(test, (test.shape[0], test.shape[1]))

    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)

    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    train_m=(train[:,0])
    test_m=(test[:,0])

    scaler_m = MinMaxScaler(feature_range=(0, 1))
    train_m=train_m.reshape(train_m.shape[0],1)

    scaler_m = scaler_m.fit(train_m)
    test_m=test_m.reshape(test_m.shape[0],1)

    scaler_d = MinMaxScaler(feature_range=(0, 1))
    scaler_d = scaler_d.fit(test_m)

    return scaler, train_scaled, test_scaled, scaler_d,scaler_m


def LSTM(code2,code3,code4,code5,work,interval):
    # fix ranrdom seed for reproducibility
    np.random.seed(7)
    dataset,dataframe_=(Create_new_added(work,code2,code3,code4,code5,interval))

    scaler,train,test,scaler_d,scaler_m=scale(dataset)
    #scaling making sure scalers for test and training sets are set correctly

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # create and fit the LSTM network
    model = Sequential()
#    model.add(Masking(mask_value=-1,input_shape=(look_back,5)))
    model.add(LSTM(units=400,input_shape=(look_back,4),return_sequences=True))
    model.add(Activation('relu'))
    model.add(LSTM(100))

    model.add(Dense(1))
    model.add(Dropout(0.001))
    #future optimization maybe do grid search? new
#    keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.fit(trainX, trainY, epochs=50, batch_size=3, verbose=1)
    # make prediction
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler_m.inverse_transform(trainPredict)
    trainY = scaler_m.inverse_transform([trainY])

    # invert tests
    testPredict = scaler_m.inverse_transform(testPredict)
    testY = scaler_m.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))

    #readjust the plot values to display
    dataset=dataset[:,0] # take the first column
    dataset=dataset.reshape(dataset.shape[0],1) #get right dimensions

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :]= np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    return dcc.Graph(
        id='lstm',
        figure={
            "data": [
                go.Scatter(
                    x=dataframe_['log_time'],
                    y=dataset,
                    mode='lines',
                    name='Real'
                ),

                go.Scatter(
                    x=dataframe_['log_time'],
                    y=trainPredict,
                    mode='lines',
                    name='Train'
                ),

                go.Scatter(
                    x=dataframe_['log_time'],
                    y=testPredict,
                    mode='lines+markers',
                   name='Prediction'
                )
            ],

            "layout": go.Layout(
                xaxis=dict(title='date'),
                yaxis=dict(title='# of times alarms 13'),
                title='LSTM prediction Alarms/Warnings'
            ),
        }
    )
