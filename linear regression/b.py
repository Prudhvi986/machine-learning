import numpy as np
import csv
import psycopg2
import math

with open('data.csv','r') as file1:
	reader  = csv.reader(file1,delimiter=',')

	xlist=[]
	for row in reader:
		xlist.append(row)
	average = []
	for i in range(3,len(xlist[0])-2):
		avg = 0.0
		for j in range(len(xlist)):
			avg += float(xlist[j][i])
		avg = avg/len(xlist)
		print(avg)
		print (i)
		print(len(xlist))
		average.append(avg)

	print(average)
	stadev = []
	for i in range(3,len(xlist[0])-2):
		dev=0.0
		for j in range(len(xlist)):
			dev = dev+((float(xlist[j][i])-average[i-3])*(float(xlist[j][i])-average[i-3]))
			dev = dev/len(xlist)
			dev = math.sqrt(dev)
		stadev.append(dev)
		print(dev)
	print(stadev)
	for i in range(3,len(xlist[0])-2):
		for j in range(len(xlist)):
			(xlist[j][i]) = (float(xlist[j][i]) - average[i-3])/stadev[i-3]

	f =  open('data1.csv','w') 
	f.write("x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11\n")
	for i in range(len(xlist)):
		for j in range(len(xlist[i])):
			if j==len(xlist[0])-1:
				f.write(str(xlist[i][j]))
			else:
				f.write(str(xlist[i][j])+ ',')
		f.write("\n")


f.close()


