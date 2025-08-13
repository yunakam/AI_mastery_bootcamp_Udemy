# map function - apply a function to all items in an iterable
# --> returns a map object (which is an iterator)
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
print("Squared numbers:", squared_numbers)


# filter function - filter items in an iterable based on a condition
# --> returns a filter object (which is an iterator)
even_list = list(filter(lambda x: x % 2 == 0, numbers)) # only even numbers are kept
print("Even numbers:", even_list)

# reduce function - apply a function cumulatively to the items of an iterable
from functools import reduce

sum_of_numbers = reduce(lambda x, y: x + y, numbers) # sum of all numbers: 1+2+3+4+5
print("Sum of numbers:", sum_of_numbers)

product_of_numbers = reduce(lambda x, y: x * y, numbers) # product of all numbers: 1*2*3*4*5 
print("Product of numbers:", product_of_numbers)