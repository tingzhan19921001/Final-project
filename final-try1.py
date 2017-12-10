import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from sklearn.metrics import mean_squared_error

class ConversionRate():
    def __init__(self, x, y, next_x):
        """
        x , y are both arrays. next_x is a number in line
        :param x:
        :param y:
        :param next_x:
        """
        self.x = x
        self.y = y
        self.next_x = next_x


def ReadcsvTodf(input_file):
    extension = os.path.splitext(input_file)[1]
    if extension == '.csv':
        df = pd.read_csv(input_file)
    elif extension == '.xlsx':
        df = pd.read_excel(input_file)

    list_years = list(df.axes[0])
    total_rows = len(list_years)

    return df,total_rows
#ReadcsvTodf()
def RegressionFit(x,y,n):
    """
    this function is to calculate the coefficients for polynomial
    :return:
    """
    return np.polyfit(x,y,n)
def FitPlot(x,p,color='b-'):
    """
    this function is to plot the fit line
    :return:
    """
    #plt.plot(x, y, 'o')
    #plt.plot(x, np.polyval(p1, x), 'r-')

    return plt.plot(x, np.polyval(p,x),color)
pass
def MSE(y_true,y_pred):
    """
    this function is to get minimal squared error to figure out the best fit
    y_true and y_pred are arrays
    :return mean squared error
    """
    mse = mean_squared_error(y_true, y_pred)

    return mse


def FitCurve(x,y):
    #df, total_rows = ReadcsvTodf('test.csv')
    #x = np.arange(1, total_rows+1 )
    #y = np.array(df['Population'])

    p1 = RegressionFit(x, y, 1)
    p2 = RegressionFit(x, y, 2)
    p3 = RegressionFit(x, y, 3)


    return p1, p2, p3
#FitCurve()

def Prediction(x,p):
    prediction = np.poly1d(p)
    prediction_array = []

    for i in x:
        prediction_array.append(prediction(i))

    return prediction_array

def FindBestFit(x,y,p1,p2,p3):

    pred1 = Prediction(x, p1)
    # print(pred1)
    pred2 = Prediction(x, p2)
    pred3 = Prediction(x, p3)

    mse1 = MSE(y, pred1)
    mse2 = MSE(y, pred2)
    mse3 = MSE(y, pred3)

    list_mse = [mse1, mse2, mse3]
    min_mse = min(list_mse)
    bestfit = list_mse.index(min_mse) + 1
    return bestfit, min_mse

def NumberofHouseholdShopWalmart():

    df, total_rows = ReadcsvTodf('test.csv')
    #df2, total_rows2 = ReadcsvTodf('test2.csv')
    #print(df)
    print(total_rows)

    x = np.arange(1,total_rows+1)

    y = np.array(df['Population'])

    p1, p2, p3 = FitCurve(x,y)
    #print(p1,p2,p3)

    bestfit, min_mse = FindBestFit(x,y,p1,p2,p3)

    print('The best fit is {} and the minimum mse is {}'.format(bestfit, min_mse))

    next_x = 20
    line = np.poly1d(p3)
    next_y_shop = line(next_x)

    print('Number of households that shopped at Walmart Supercenter grocery stores \n within the last 7 days in the United States from spring 2008 to spring 2017 (in millions)',next_y_shop)
    #print(population)
    return next_y_shop

#NumberofHouseholdShopWalmart()

def NumberofHousehold():
    """
    This function will get how many Number of household in US.

    :return:
    """
    df, total_rows = ReadcsvTodf('test_household.xlsx')
    #list_years = list(df.axes[0])
    #total_rows = len(list_years)
    print(total_rows)

    x = np.arange(1, total_rows + 1)

    y = np.array(df['Household'])

    p1, p2, p3 = FitCurve(x, y)

    bestfit, min_mse = FindBestFit(x,y,p1,p2,p3)
    print('The best fit is {} and the minimum mse is {}'.format(bestfit, min_mse))

    next_x = 49
    line = np.poly1d(p3)
    next_y_household = line(next_x)
    print('In 2017 the number of household in US is(in millions): ',next_y_household)
    #x = np.arange(1,total_rows+1)

    return next_y_household
NumberofHousehold()

def ConversionRate(a, b):

    rate = a/7/b
    print(rate)

    return rate

def Customer():

    c = np.random.poisson(1.03, 1000)
    count,bins,ignored = plt.hist(c,9,normed = True)
    plt.show()

#Customer()
