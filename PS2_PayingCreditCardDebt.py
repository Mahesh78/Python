## Paying Off Credit Card Debt

# Part 1: Paying the Minimum

TotalPaid = 0

for i in range(1,13):
    MinPay = round(balance * monthlyPaymentRate,2)
    NewBal = round(balance - MinPay,2)
    NewBal += round(NewBal*(annualInterestRate/12),2) #UnpaidBal = (balance - MinPay)*(annualInterestRate/12)+(balance - MinPay)
    
    print 'Month: ' + str(i)
    print 'Minimum monthly payment: '+str(MinPay)
    print 'Remaining balance: '+str(NewBal)
    
    balance = NewBal
    TotalPaid += MinPay
print 'Total paid: '+str(TotalPaid)
print 'Remaining balance: '+str(NewBal)


# Part 2: Paying Debt Off In a Year

MinPay = 0
NewBal = 1
while NewBal > 0:
    MinPay += 10
    NewBal = balance

    for i in range(1,13):    
        NewBal = round(NewBal - MinPay,2)
        NewBal += round(NewBal*(annualInterestRate/12.0),2)
        
print 'Lowest Payment: ' + str(MinPay)


# Part 3: Using Bisection Search to Make the Program Faster 

epsilon = 0.01

LowerBound = balance/12
HigherBound = (balance*(1+annualInterestRate)**12)/12

NewBal = balance
MinPay = (LowerBound + HigherBound)/2

for i in range(1,13):
    
    NewBal = NewBal - MinPay
    NewBal += NewBal*(annualInterestRate/12.0)

    
while abs(NewBal) > epsilon:
    if NewBal > 0:
        LowerBound = MinPay
    else:
        HigherBound = MinPay
    NewBal = balance
    MinPay = (LowerBound + HigherBound)/2
    for i in range(1,13):
    
        NewBal = NewBal - MinPay
        NewBal += NewBal*(annualInterestRate/12.0)
    
print 'Lowest Payment: '+ str(round(MinPay,2))