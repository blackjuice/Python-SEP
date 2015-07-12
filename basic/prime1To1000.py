#!/usr/bin/python
# execution: prime1To1000.py

# trivial solution with print
array = [] # declaring array
y = 0
# storing in array
for i in range(2, 1000):
    prime = 1
    for j in range(2, i):
        if i % j == 0:
            prime = 0
            break
        prime = 1
    if prime:
        array.append(i)
        # apparently this doesnt work:
        #   array[y] += i ,which would be the same as array[y] = i
        #   array += i, nor this

print "result stored in an array"
print array
print "there are", len(array), "prime numbers"
