import math
print math.log(1)
print math.log(0.00000000000000000001)
print math.log(0.5)


a = 0.8
ans = 0.8

for i in range(13):
    print "After %s months = %.0f%%" % (i, (100*(1-math.pow(a,i))))