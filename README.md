
# Title: 
Forecasting Daily Supermarket Profit Using Monte Carlo Simulation

## Team Member(s):
Wenjun Ma 
Ting Zhan

# Monte Carlo Simulation Scenario & Purpose:

Scenario: 
In this project, we are trying to simulate the daily profit made by a local supermarket. The supermarket we refer to are markets like Meijer or Walmart. In our daily life, we go to shopping in supermarkets at least once a week or more. Predicting the daily profit in different situations helps the supermarkets to prepare well.

Here we adopt a top down method to predict daily supermarket profit. 
First we start with: Profit(P) = Income(I) - Expenses(E). It is true that both income and expense will change on daily basis, especially holiday season vs non-holiday season. Just like the past Thanksgiving holiday and the coming Christmas holiday, the profit is tremendously from other days.

Therefore, we have separated into two situations when predicting the profit: Holiday vs Non-Holiday.

Secondly, as in real world, the income comes from the accumulation of each transaction. 
Therefore, Income = Number of Sales(S) * Profit per Sale(P). 

Thirdly, Number of Sales(S) = Volume of Customer(C) * coversion rate (R).

Fourthly, the Expenses is a combination of fixed overhead (H) plus the total cost of the conversion.

Fifthly, the conversion rate is fluenced by the marketing channels, including TV, Radio and Advertisement. The cost of a conversion varies between $0.20 and $0.80.

The final model is: P = C*R*S - 





### Hypothesis before running the simulation:

1. Income and Expense are independent. The calculation of income will not be influenced by expense.

2. The profit per sale is calculated by transacation amount mulitplied by a certain percentage. The reason why we build model like this is that we beleive that the list price of each item in the supermarket is set at the cost price multiplied by 1.30, which we hypothesize that the list price is 30% more than the cost price.

3. The Expenses is a combination of fixed overhead (H) plus the total cost of the leads. This amount would come from average expenses for each supermarket in real world.

4. The number of sales per day is the number of customer multiplied by the conversion rate. The conversion rate is hypothesized as 0.85 percent. It will be influenced by how much it  modified if we find more about this.

5. The cost of a conversion varies between $0.20 and $0.80. In calculation, we will take the average value for the cost. That is $0.60.

### Simulation's variables of uncertainty
List and describe your simulation's variables of uncertainty (where you're using pseudo-random number generation). 
For each such variable, how did you decide the range and which probability distribution to use?  
Do you think it's a good representation of reality?

## Instructions on how to use the program:


## Sources Used:

