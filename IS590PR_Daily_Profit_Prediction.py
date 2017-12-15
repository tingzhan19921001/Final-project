"""
IS590PR Final Project

This function works as below:
User can input the city and state, for example, urbana,il or chicago,il to do a local prediction.
The function will give a prediction about the daily profit range and the percentage each range takes.
A bar chart is also created to help visualization.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from sklearn.metrics import mean_squared_error
import requests
import re
from datetime import datetime

class ConversionRate():
    def __init__(self,input_flie):
        """

        :param input_flie: data file dowloaded online to fulfill prediction
        """
        self.input_file = input_flie



    def ReadcsvTodf(self):
        """
        Input file might have different extensions, normally they csv or xlsx. therefore, we need different methods to
        read the data file based on their extensions
        :return:
        """
        input_file = self.input_file
        extension = os.path.splitext(input_file)[1] # fetch the extension here
        if extension == '.csv':
            df = pd.read_csv(input_file)
        elif extension == '.xlsx':
            df = pd.read_excel(input_file)

        list_years = list(df.axes[0]) # to get how many rows
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
        this function is to plot the fit line, if you wish to see the linear regression line, you can employee this
        function. But I do not plot the fitline in my diagram. If you wish to, you could plot it.
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
        >>> ConversionRate.MSE([2,3,4,5],[3,4,5,6])
        1.0
        >>> ConversionRate.MSE(np.arange(1,10),np.arange(15,24))
        196.0
        """
        mse = mean_squared_error(y, y_pred) # both y and y_pred are arrays

        return mse


    def FitCurve(self,x,y):
        """

        :param x: the x are the x-axis from the data, normally they are years
        :param y: the actual value in that year. All from the data file downloaded online
        :return:

        """

        p1 = self.RegressionFit(x, y, 1)
        p2 = self.RegressionFit(x, y, 2)
        p3 = self.RegressionFit(x, y, 3)


        return p1, p2, p3
    #FitCurve()

    @staticmethod
    def Prediction(x,p):
        """

        :param x: the x are the x-axis from the data, normally they are years
        :param p: p are the co-efficients calculated  in Fitcurve
        :return: a prediction array
        I put all the prediction values into an array because I want to use it to calculate the mean squared errors
        """
        prediction = np.poly1d(p)
        prediction_array = []

        for i in x:
            prediction_array.append(prediction(i))


        return prediction_array

    def FindBestFit(self, x,y,p1,p2,p3):
        """

        :param x:
        :param y:
        :param p1:
        :param p2:
        :param p3:
        :return:
        In this function,  I want to find the smallest mean squared errors which helps to determine the best fit
        """

        pred1 = self.Prediction(x, p1)
        pred2 = self.Prediction(x, p2)
        pred3 = self.Prediction(x, p3)
        # to get three arrays using p1,p2,p3

        list_p_coefficients = [p1,p2,p3]

        mse1 = self.MSE(y, pred1)
        mse2 = self.MSE(y, pred2)
        mse3 = self.MSE(y, pred3)
        # to calculate the MSE between the real y and three y-predictions generated from the last step

        list_mse = [mse1, mse2, mse3]
        min_mse = min(list_mse)
        bestfit = list_p_coefficients[list_mse.index(min_mse)]
        # find the smallest mse, and determine the best p

        return bestfit, min_mse

    def NextFigure(self):
        """

        :return: the next y predicted using the real data
        """

        df, total_rows = self.ReadcsvTodf()

        x = np.arange(1,total_rows+1)

        next_x = len(x)+1

        y = np.array(df['Population'])

        p1, p2, p3 = self.FitCurve(x,y)

        bestfit, min_mse = self.FindBestFit(x,y,p1,p2,p3)

        line = np.poly1d(bestfit)

        next_y = line(next_x)

        return next_y
#ConversionRate('data_shoppers.csv').NextFigure()

def RateCalculation(a, b):

    """

    :param a: number of total household(in millions) who shopped at Walmart in the last 7 days therefore
    a divided by 7 equals to the number of household in one day
    :param b: number of total household(in millions) in the nation
    :return: the conversion rate through out the country
    >>> RateCalculation(700,1000)
    0.1
    >>> RateCalculation(2100,6000)
    0.05
    """
    rate = a/7/b

    return rate


class Customer():
    def __init__(self,lam,size):
        self._lam = lam
        self._size = size

    def Customer_distribution(self):
        """

        :return: the self._lam would be using the local people times the conversion rate
        """

        cust = np.random.poisson(self._lam,self._size)
        return cust
#Customer()


def DataFetch():
    """

    :return: this function is to fetch from online the real data of population and household income.
    We believe that the median household income and the weekday vs non-weekday will influence the conversion rate
    """
    while True:
        """
        this while true step is added to prevent input error.
        For example, if users input urbana, ny, and we actually dont have a city named urbana
        in new york state. therefore, the program will ask you to re enter your location and 
        
        """
        try:
            city_state = input("What is your city and state: (example: urbana,il)").split(',')
            # tell me where you are

            url = 'https://datausa.io/profile/geo/'+'{0}-{1}/'.format(city_state[0],city_state[1])
            # generate the url to for the input location

            res = requests.get(url)

            res = res.text.encode(res.encoding).decode('utf-8')

            find_pop = re.findall(r'\S*(pop\&rank)\S*>(.*?)</span>',res)[0]
            #to fetch the population value

            find_median_income = re.findall(r'\S*(income\&rank)\S*>(.*?)</span>',res)[0]
            #to fetch the median household income

            break


        except IndexError:
            print('The city or state is not recognized, please re-enter.')
            continue


    population = find_pop[1].replace(',','') # change the datatype
    population=int(population)

    median_household_income = str(find_median_income[1]).strip('$').replace(',','') #change the datatype
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
        input rent value
        :return:
        """
        return self.rent

    @property
    def utility_bills(self):
        """
        input supermarket daily utility_bills
        :return:
        """
        return self.utility_bills


    @property
    def insurance(self):
        """
        input supermarket insurance cost
        :return:
        """
        return self.insurance

    @property
    def technology(self):
        """
        input supermarket daily technology cost
        :return:
        """
        return self.technology

    @property
    def marketing(self):
        """
        input supermarket marketing cost
        :return:
        """
        return self.marketing

    @property
    def salaries(self):
        """
        input supermarket salaries cost
        :return:
        """
        return self.salaries


class profit_persale():
    def __init__(self,size):
        """
        initialize profit_persale with distribution size
        :param size:
        :return:
        """
        self.size = size

    def profit(self):
        """
        Assume the profit_persale is uniform distribution
        we set the minimum profit 10 and the maximun 200
        :return:
        """
        self.profit_random = np.random.uniform(10,200,self.size)
        return self.profit_random

class conversion_cost():
    def __init__(self,mu,sigma,low,high):
        """
        initialize conversion_cost class with mu,sigma,low,high
        :param mu:
        :param sigma:
        :param low:
        :param high:
        """
        self._mu = mu
        self._sigma = sigma
        self._low = low
        self._high = high

    def conversioncost_random(self):
        """
        Assume the conversion cost is normal distribution
        :return:
        """
        cost_random= np.random.normal(self._mu,self._sigma,1000)# 1000 is the size we set for this random distribution

        return cost_random



def ProfitDistribution(list_profit):

    """

    :param list_profit: The daily profit prediction(1000 values)
    :return:
    """
    small_than_100000 = 0
    between_100000_to_300000 = 0
    between_300000_to_500000 = 0
    between_500000_to_700000 = 0
    larger_than_700000 = 0
    len_profit = len(list_profit)
    perc1 = 0
    perc2 = 0
    perc3 = 0
    perc4 = 0
    perc5 = 0

    for i in list_profit:
        if i < 100000:
            small_than_100000 += 1
            perc1 = small_than_100000/len_profit

        elif 100000 <= i <=300000:
            between_100000_to_300000 += 1
            perc2 = between_100000_to_300000/len_profit

        elif 300000 <= i <= 500000:
            between_300000_to_500000 += 1
            perc3 = between_300000_to_500000/len_profit

        elif 50000 <= i <= 700000:
            between_500000_to_700000 += 1
            perc4 = between_500000_to_700000/len_profit

        else:
            larger_than_700000 += 1
            perc5 = larger_than_700000/len_profit

    return perc1,perc2,perc3,perc4,perc5

def BarchartPlot(list_profit):
    """

    :param data: the percentage of daily profit prediction and are fetched from ProfitDistribution()
    :return: a bar chart
    """

    objects = ('<100', '100~300', '300~500', '500~700', '>700')
    y_pos = np.arange(len(objects))
    perc1, perc2, perc3, perc4, perc5 = ProfitDistribution(list_profit)

    percentage = [perc1, perc2, perc3, perc4, perc5]

    plt.bar(y_pos, percentage, align='center', alpha=0.5, width = 0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage')
    plt.xlabel('Daily Profit Range (in thousands)')
    plt.title('Daily Profit Prediction Distribution')

    plt.show()

def main():
    local_population, local_income = DataFetch() # get local pupulation, local income from website.

    People_shopping = ConversionRate('data_shoppers.csv').NextFigure()
    People_total = ConversionRate('data_household.xlsx').NextFigure()

    conversion_rate = RateCalculation(People_shopping,People_total).item()

    US_Household_Income_median = 59039 #this is fetched online

    dayOfWeek = datetime.now().weekday()  # show today date between 0~6 represent sunday to monday

    if local_income > US_Household_Income_median : #if local income larger than average income in country

        if dayOfWeek >4: #if in the weekend, the conversion rate will be bigger
            conversion_rate = conversion_rate * 1.5

        else: #else in the weekday
            conversion_rate = conversion_rate * 1.2

    elif local_income<= US_Household_Income_median:

        if dayOfWeek>4:
            conversion_rate = conversion_rate * 1.1

        else:
            conversion_rate = conversion_rate * 0.8

    over = overhead(3000,200,300,400,500,6000) #set overhead value

    profit_distribution = profit_persale(1000).profit()# get profit_persale distribution,the size is 1000

    conversioncost = conversion_cost(0.5,1.0,0.2,0.8) # get conversioncost distribution

    customerdistribution = Customer(local_population*conversion_rate,1000).Customer_distribution()
    # get customer distributiuon, the size is 1000

    Expense = over._rent +over._utility_bills + over._insurance + over._technology + over._marketing +over._salaries+customerdistribution*conversioncost.conversioncost_random()

    Income = customerdistribution * profit_distribution

    DailyProfit = Income - Expense

    perc1, perc2, perc3, perc4, perc5 = ProfitDistribution(DailyProfit)

    print('The percentage of profit distribution:\n <100000: {0:.2f}% \n 100000~300000: {1:.2f}% \n 300000~500000:{2:.2f}% \n 500000~700000 : {3:.2f}% \n >700000 : {4:.2f}%'.format(perc1, perc2, perc3, perc4, perc5))
    print('A bar chart is created behind this screen!')
    BarchartPlot(DailyProfit) # if you do not wish to see the bar chart, you can comment this line


main()
