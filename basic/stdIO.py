import sys

sumOp = 0

for line in sys.stdin:
    num = float(line)
    sumOp = sumOp + num

print(sumOp)
