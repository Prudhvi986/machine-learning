import numpy as np
import csv
import psycopg2

with open('linregdata.csv','r') as file1:
	reader  = csv.reader(file1,delimiter=',')

	xlist = []
	for row  in reader:
		if row[0] == 'M':
			row[0] = str('1,0,0')
		elif row[0] == 'I':
			row[0] = str('0,0,1')
		elif row[0] == 'F':
			row[0] = str('0,1,0')

		xlist.append(row)
print(len(xlist))
f =  open('data.csv','w') 
f.write("x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11\n")
for i in range(len(xlist)):
	for j in range(len(xlist[i])):
		if j==len(xlist[0])-1:
			f.write(str(xlist[i][j]))
		else:
			f.write(str(xlist[i][j])+ ',')
	f.write("\n")

f.close()
		


	#print(reader)