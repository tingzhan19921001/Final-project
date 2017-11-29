
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

Therefore, we have developed two scenarios when predicting the profit: Holiday vs Non-Holiday.

Secondly, as in real world, the income comes from the accumulation of each transaction. 
Therefore, Income = Number of Sales(S) * Profit per Sale(p). 

Thirdly, Number of Sales(S) = Volume of Customer(C) * coversion rate (R).

Fourthly, the Expenses is a combination of fixed overhead (H) plus the total cost of the conversion.

Fifthly, the conversion rate is fluenced by the marketing channels, including TV, Radio and Advertisement. The cost of a conversion (c) varies between $0.20 and $0.80.

Income = C*R*S*p
Expense = H + C*R*c
The final model is: P = C*R*S*p - (H + C*R*c)

### Hypothesis before running the simulation:

1. Income and Expense are independent. The calculation of income will not be influenced by expense.

2. The profit per sale is calculated by transacation amount mulitplied by a certain percentage. For example, we thinks that the list price of each item in the supermarket is set at the cost price multiplied by a percentage, which we hypothesize that, for example, the list price is 30% more than the cost price. However,the percentage is not a fixed value because it will vary in different scenario, different supermarket, different transactions. 

3. The Expenses is a combination of fixed overhead (H) plus the total cost of each conversion. This amount would come from average expenses for each supermarket in real world.

4. The number of sales per day is the number of customer multiplied by the conversion rate. The conversion rate is hypothesized as 0.85 percent. It will be influenced by how much it  modified if we find more about this.

Before running the simulation, the value set above is for Non-Holiday scenario for testing our code. In the future, the value for Holiday scenario will be added.

### Simulation's variables of uncertainty

The variables of uncertainty are:
Volume of Customer(C): Volume of Customer will vary in several situations. For example, Holiday and Non-Holiday, different types of supermarket like Walmart vs Aldi.

Profit per Sale(p): Profit per sale will be influenced by a combination of factors. For example, how the list price it set will influence the profit per sale, just like what we discussed above. Also, as what we find in the real world, the length of stay will influence the profit. For example, if he/she tends to spend more time in supermarket, he/she will spend more in one transaction.

## Instructions on how to use the program:


## Sources Used:
This is where we get our basic idea.
https://www.vertex42.com/ExcelArticles/mc/SalesForecast.html

