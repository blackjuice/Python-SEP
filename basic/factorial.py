# First actual contact with Python
# Simple Factorial algorithm
# date: 12 July 2015

# Source: http://www.programiz.com/python-programming/examples/factorial
# modified by blackjuice

# execution: factorial.py x
#   where x is the input number

# take input from the user: num = int( input("Factorial of: ") )
from sys import argv

# recursive factorial function
def factorialR(n):
    if n == 0:
        return 1
    else:
        return n * factorialR (n - 1)

script, arg_num = argv
num = int(arg_num) # parsing string to int

factorial = 1

if num < 0:
    print "Sorry, factorial does not exist for negative numbers"
    exit()

elif num == 0:
   print "Iterative:", num, "! = 1"
else:
   for i in range(1,num + 1):
       factorial = factorial * i
   print "Iterative:", num, "! = ", factorial
 
print "Recursive:", num, "! =", factorialR(num)
