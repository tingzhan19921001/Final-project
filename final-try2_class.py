import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from sklearn.metrics import mean_squared_error
import requests
import re
from datetime import datetime

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
        this function is to plot the fit line, if you wish to see the linear regression line, you can employee this function.
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
        1
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

        y = np.array(df['Population'])

        p1, p2, p3 = self.FitCurve(x,y)


        bestfit, min_mse = self.FindBestFit(x,y,p1,p2,p3)

        line = np.poly1d(bestfit)
        next_y_shop = line(self.next_x)

        return next_y_shop


def RateCalculation(a, b):

    """

    :param a: number of total household(in millions) who shopped at Walmart in the last 7 days therefore
    a divided by 7 equals to the number of household in one day
    :param b: number of total household(in millions) in the nation
    :return: the conversion rate through out the country
    >>> RateCalculation(700,1000)
    0.1
    """
    rate = a/7/b
    #print(rate)

    return rate


class Customer():
    def __init__(self,lam,size):
        self._lam = lam
        self._size = size
    def Customer_distribution(self):

        cust = np.random.poisson(self._lam,self._size)
        return cust
#Customer()


def DataFetch():
    city_state = input("What is your city and state: (example: urbana,il)").split(',')
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
        x= np.random.normal(self._mu,self._sigma,100)
        return x

def PercentageCalculation(a,b):
    """

    :param a: divisor
    :param b: dividende
    :return: quotient
    """
    return a/b

def ProfitDistribution(list_profit):
    small_than_100000 = 0
    between_100000_to_300000 = 0
    between_300000_to_600000 = 0
    between_600000_to_900000 = 0
    larger_than_900000 = 0
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

        elif 300000 <= i <= 600000:
            between_300000_to_600000 += 1
            perc3 = between_300000_to_600000/len_profit

        elif 60000 <= i <= 900000:
            between_600000_to_900000 += 1
            perc4 = between_600000_to_900000/len_profit

        else:
            larger_than_900000 += 1
            perc5 = larger_than_900000/len_profit

    #print('The percentage of profit distribution:\n <100000: {0:.2f}% \n 100000~300000: {1:.2f}% \n 300000~600000:{2:.2f}% \n 600000~900000 : {3:.2f}% \n >900000 : {4:.2f}%'.format(perc1,perc2,perc3,perc4,perc5))

    return perc1,perc2,perc3,perc4,perc5

def BarchartPlot(list_profit):
    """

    :param data: the percentage of daily profit prediction and are fetched from ProfitDistribution()
    :return: a bar chart
    """

    objects = ('<100', '100~300', '300~600', '600~900', '>900')
    y_pos = np.arange(len(objects))
    perc1, perc2, perc3, perc4, perc5 = ProfitDistribution(list_profit)

    percentage = [perc1, perc2, perc3, perc4, perc5]

    plt.bar(y_pos, percentage, align='center', alpha=0.5, width = 0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage')
    plt.xlabel('Daily Profit (in thousands)')
    plt.title('Daily Profit Prediction Distribution')

    plt.show()

def main():
    local_population, local_income = DataFetch() # get local pupulation, local income from website.
    People_shopping = ConversionRate('test.csv',20).NextFigure()
    People_total = ConversionRate('test_household.xlsx',49).NextFigure()
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

    profit_distribution = profit_persale(100).profit()# get profit_persale distribution
    conversioncost = conversion_cost(0.5,1.0,0.2,0.8) # get conversioncost distribution
    customerdistribution = Customer(local_population*conversion_rate,100).Customer_distribution() # get customer distributiuon
    # expense = overhead + customer*conversion_cost
    Expense = over._rent +over._utility_bills + over._insurance + over._technology + over._marketing +over._salaries+customerdistribution*conversioncost.conversioncost_random()
    # Income = local_people*conversion_rate*profit_persale = customer * profit_persale
    Income = customerdistribution * profit_distribution
    DailyProfit = Income - Expense

    perc1, perc2, perc3, perc4, perc5 = ProfitDistribution(DailyProfit)
    print('The percentage of profit distribution:\n <100000: {0:.2f}% \n 100000~300000: {1:.2f}% \n 300000~600000:{2:.2f}% \n 600000~900000 : {3:.2f}% \n >900000 : {4:.2f}%'.format(perc1, perc2, perc3, perc4, perc5))

    BarchartPlot(DailyProfit)


main()