import numpy as np
from datetime import datetime,date
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
        self.profit_random = np.random.uniform(0,500)
        return self.profit_random

class conversion_cost():
    def __init__(self,mu,sigma,low,high):
        self._mu = mu
        self._sigma = sigma
        self._low = low
        self._high = high

    def co_random(self):
        x = self._low -1
        while x < self._low or x > self._high:
            x = np.random.normal(self._mu,self._sigma)
        return x




over = overhead(1,2,3,4,5,6)
over._rent = 1
over._utility_bills = 2
over._insurance = 3
over._technology = 4
over._marketing = 5
over._salaries =6
p = profit_persale(1000)
p.profit()
con = conversion_cost(0.5,1.0,0.2,0.8)
con.co_random()
expense = over._rent +over._utility_bills + over._insurance + over._technology + over._marketing +over._salaries+p.profit() * con.co_random()
dayOfWeek = datetime.now().weekday()


'''
if a>GDP :
    if dayOfWeek >4:
        conversion_weekday = conversion_rate * 0.15
    else:
        conversion_weekend = conversion_rate * 0.12
elif a<=GDP:
    if dayOfWeek<=4:
        conversion_weekday = conversion_rate * 0.11
    else:
        conversion_weekend = conversion_rate * 0.8
        '''




























