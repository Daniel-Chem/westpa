import sys
fac = float(sys.argv[1])
answer = 1
while fac > 1:
    answer *= fac
    fac -= 1
print(answer)