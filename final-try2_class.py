import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from sklearn.metrics import mean_squared_error
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime,date

class ConversionRate():
    def __init__(self,input_flie, next_x):
        """
        x , y are both arrays. next_x is a number in line
        :param x:
        :param y:
        :param next_x:
        """
        self.input_file = input_flie
        self.next_x = next_x


    def ReadcsvTodf(self):
        input_file = self.input_file
        extension = os.path.splitext(input_file)[1]
        if extension == '.csv':
            df = pd.read_csv(input_file)
        elif extension == '.xlsx':
            df = pd.read_excel(input_file)

        list_years = list(df.axes[0])
        total_rows = len(list_years)

        return df,total_rows
    #ReadcsvTodf()
    @staticmethod
    def RegressionFit(x,y,n):
        """
        this function is to calculate the coefficients for polynomial
        :return:
        """
        return np.polyfit(x,y,n)

    @staticmethod
    def FitPlot(x,p,color='b-'):
        """
        this function is to plot the fit line
        :return:
        """
        #plt.plot(x, y, 'o')
        #plt.plot(x, np.polyval(p1, x), 'r-')

        return plt.plot(x, np.polyval(p,x),color)
    pass

    @staticmethod
    def MSE(y,y_pred):
        """
        this function is to get minimal squared error to figure out the best fit
        y_true and y_pred are arrays
        :return mean squared error
        """
        mse = mean_squared_error(y, y_pred)

        return mse


    def FitCurve(self,x,y):
        #df, total_rows = ReadcsvTodf('test.csv')
        #x = np.arange(1, total_rows+1 )
        #y = np.array(df['Population'])

        p1 = self.RegressionFit(x, y, 1)
        p2 = self.RegressionFit(x, y, 2)
        p3 = self.RegressionFit(x, y, 3)


        return p1, p2, p3
    #FitCurve()

    @staticmethod
    def Prediction(x,p):
        prediction = np.poly1d(p)
        prediction_array = []

        for i in x:
            prediction_array.append(prediction(i))

        return prediction_array

    def FindBestFit(self, x,y,p1,p2,p3):

        pred1 = self.Prediction(x, p1)
        # print(pred1)
        pred2 = self.Prediction(x, p2)
        pred3 = self.Prediction(x, p3)

        mse1 = self.MSE(y, pred1)
        mse2 = self.MSE(y, pred2)
        mse3 = self.MSE(y, pred3)

        list_mse = [mse1, mse2, mse3]
        min_mse = min(list_mse)
        bestfit = list_mse.index(min_mse) + 1
        return bestfit, min_mse

    def NextFigure(self):

        df, total_rows = self.ReadcsvTodf()

        x = np.arange(1,total_rows+1)
        #print(x)
        y = np.array(df['Population'])

        p1, p2, p3 = self.FitCurve(x,y)
        #print(p1,p2,p3)

        bestfit, min_mse = self.FindBestFit(x,y,p1,p2,p3)

        #print('The best fit is {} and the minimum mse is {}'.format(bestfit, min_mse))

        #next_x = 20
        line = np.poly1d(p3)
        next_y_shop = line(self.next_x)

        #print('Number of households that shopped at Walmart Supercenter grocery stores \n within the last 7 days in the United States from spring 2008 to spring 2017 (in millions)',next_y_shop)
        #print(population)
        return next_y_shop

    #NumberofHouseholdShopWalmart()




def RateCalculation(a, b):

    """

    :param a: number of total household(in millions) who shopped at Walmart in the last 7 days therefore
    a divided by 7 equals to the number of household in one day
    :param b: number of total household(in millions) in the nation
    :return: the conversion rate through out the country
    """
    rate = a/7/b
    #print(rate)

    return rate


class Customer():
    def __init__(self,lam,size):
        self._lam = lam
        self._size = size
    def Customer_distribution(self):

        c = np.random.poisson(self._lam,self._size)
        #count,bins,ignored = plt.hist(c,9,normed = True)
        return c
#Customer()


def DataFetch():
    city_state = input("What is your city and state").split(',')
    url = url = 'https://datausa.io/profile/geo/'+'{0}-{1}/'.format(city_state[0],city_state[1])
    res = requests.get(url)
    res = res.text.encode(res.encoding).decode('utf-8')




    find_pop = re.findall(r'\S*(pop\&rank)\S*>(.*?)</span>',res)[0]
    find_median_income = re.findall(r'\S*(income\&rank)\S*>(.*?)</span>',res)[0]

    #print(find_pop[1])
    population = find_pop[1].replace(',','')
    population=int(population)
    median_household_income = str(find_median_income[1]).strip('$').replace(',','')
    median_household_income = int(median_household_income)
    #print(median_household_income)
    return population,median_household_income



class overhead():
    def __init__(self,rent,utility_bills,insurance,technology,marketing,salaries ):
        self._rent = rent
        self._utility_bills = utility_bills
        self._insurance = insurance
        self._technology = technology
        self._marketing = marketing
        self._salaries = salaries
    @property
    def rent(self):
        """

        :return:
        """
        return self.rent

    @property
    def utility_bills(self):
        """

        :return:
        """
        return self.utility_bills


    @property
    def insurance(self):
        """

        :return:
        """
        return self.insurance

    @property
    def technology(self):
        """

        :return:
        """
        return self.technology

    @property
    def marketing(self):
        """

        :return:
        """
        return self.marketing

    @property
    def salaries(self):
        """

        :return:
        """
        return self.salaries


class profit_persale():
    def __init__(self,size):
        """

        :param size:
        :return:
        """
        self.size = size

    def profit(self):
        """

        :return:
        """
        self.profit_random = np.random.uniform(0,500,self.size)
        return self.profit_random

class conversion_cost():
    def __init__(self,mu,sigma,low,high):
        self._mu = mu
        self._sigma = sigma
        self._low = low
        self._high = high

    def co_random(self):
        x= np.random.normal(self._mu,self._sigma,100)
        return x

local_population, local_income = DataFetch()
People_shopping = ConversionRate('test.csv',20).NextFigure()
People_total = ConversionRate('test_household.xlsx',49).NextFigure()

conversion_rate = RateCalculation(People_shopping,People_total).item()

print('!!!',type(conversion_rate))


dayOfWeek = datetime.now().weekday()
US_Household_Income_median = 59039

if local_income > US_Household_Income_median :
    if dayOfWeek >4:
        conversion_rate = conversion_rate * 1.5
    else:
        conversion_rate = conversion_rate * 1.2
elif local_income<= US_Household_Income_median:
    if dayOfWeek>4:
        conversion_rate = conversion_rate * 1.1
    else:
        conversion_rate = conversion_rate * 0.8


over = overhead(1,2,3,4,5,6)
over._rent = 1
over._utility_bills = 2
over._insurance = 3
over._technology = 4
over._marketing = 5
over._salaries =6
profit_distribution = profit_persale(100).profit()
con = conversion_cost(0.5,1.0,0.2,0.8)


cust = Customer(local_population*conversion_rate,100).Customer_distribution()

print(cust)
expense = over._rent +over._utility_bills + over._insurance + over._technology + over._marketing +over._salaries

print(type(local_population))
print(type(conversion_rate))
print(type(con.co_random()))

ex1= con.co_random()
print(ex1)

print(profit_distribution)

Income = cust * profit_distribution
#DailyProfit = Income - expense-ex1
print(Income)