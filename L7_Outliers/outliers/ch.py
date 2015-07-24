x = [2,3,4,5,6]
y= [4,56,6,7,8]
n = [3,5,6,76,7]
l = len(x)
c = 0
e = []
for i in range(l):
	c =abs(x[i] - n[i])
	e.append(c)
	i = i + 1
zipped = zip(x,n,e)
print zipped
nw = []

zipped.sort(key=lambda tup: tup[2])
i = 0
for i in range(len(zipped)*(80)/100):
	w = zipped[i]
	nw.append(w)
	i = i + 1

print nw

