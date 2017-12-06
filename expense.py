import numpy as np
import pandas as pd
class overhead():
    def _init_(self,rent=0,utility_bills=0,insurance=0,technology=0,marketing=0,salaries=0 ):
        self.rent = rent
        self.utility_bills = utility_bills
        self.insurance = insurance
        self.technology = technology
        self.marketing = marketing
        self.salaries = salaries
    @property
    def rent(self):
        return self.rent

    @property
    def utility_bills(self):
        return self.utility_bills


    @property
    def insurance(self):
        return self.insurance

    @property
    def technology(self):
        return self.technology

    @property
    def marketing(self):
        return self.marketing

    @property
    def salaries(self):
        return self.salaries

class profit_perhour():

    def _init_(self,max_time):
        self.max_time = max_time

    def profit(self,time_end):
        self.time = np.arrange(0,time_end)
        return self.time

    def time_dis(self):
        self.time_dis = stats.poisson.pmf(self.time, self.max_time )










