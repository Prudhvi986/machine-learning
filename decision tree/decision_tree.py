# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import math
import numpy as np

# Enter You Name Here
myname = "Prudhvi" # or "Doe-Jane-"

def entropy(data) :
    #print "data"
    if len(data) ==0:
        return 0
    values = list(row[-1] for row in data)
    if len(values) == 0:
        return 0 
    #print values
    class_values =['1','0']
    #print class_values
   # print values
   # print len(values)
    #print class_values[0]
    #print class_values[1]
    t=0.0
    f=0.0
    for i in values:
        if i == class_values[0]:
            t=t+1
        else:
            f=f+1
    #print"total"
    #print len(values)
    #print"true"
    #print t
    #print"false"
    #print f
    p = t/len(values)
    #print p
    if p==0 or p==1:
        return 0.0
    entropy_value = -(p*math.log(p,2) + (1-p)*math.log((1-p),2))
    return entropy_value
    

def test_split(data,value,col) :
    right = list()
    left = list()
    for row in data:
        if row[col] >value:
            right.append(row)
        else :
            left.append(row)
    return right,left    

def best_split(data):
    class_values = list(set(row[-1] for row in data))
    #print class_values
    #print data[0]
    info_gain,attr,attr_value = 0.0,0,0.0
    for i in range(len(data[0])-1) :
       # print"total length"
       # print len(data[0])-1
        
       #     print i
        unique_set = list((row[i] for row in data))
        unique_set = list(set(unique_set))
     
        for j  in range(len(unique_set)):
            rt,lt = test_split(data,unique_set[j],i)
            p = len(rt)/float(len(rt)+len(lt))
            check_value = entropy(data)-((1-p)*entropy(lt)+p*entropy(rt)) 
            #print "checkvalue"
            #print check_value
            #print "infogain"
            #print info_gain
            if check_value > info_gain:
                #print" kshdfsdgfdsbfdskjhdsfdshfjdsfidsfdsfjadsfsjbf"
                info_gain = check_value
                attr = i;
                attr_value = unique_set[j]
        #print "attr"
        #print attr
        #print "attr_value"
        #print attr_value      
    return attr,attr_value,info_gain


                       

def build_tree(data):
    #print "fuck you"
    #print len(data)
    class_values = ['1','0']
    
    
    values = list(row[-1] for row in data)
    attr,attr_value,info_gain = best_split(data)
    #print entropy(data)
    if info_gain <=0 :
        p = values.count(class_values[0])
        if p > (len(data) -p):
            end = Leaf(class_values[0],data)
        else:
            end = Leaf(class_values[1],data)
        return end
    
    print ("attr:")
    print (attr)
    print ("attr_value:")
    print (attr_value)
    rt,lt = test_split(data,attr_value,attr)
    #print len(rt)
    #print len(lt)
    right = build_tree(rt)
    #print "right"
    left = build_tree(lt)
    #print "left"
    mynode = Node(attr,attr_value,data,rt,lt,right,left) 
   # print "mynode"  
    return mynode

def predict(mynode,test_inst):
    if isinstance(mynode,Leaf):
        return mynode.label
    if test_inst[mynode.attr] < mynode.attr_value:
        if isinstance(mynode.left,Leaf) :
            return mynode.left.label
        else:
            return predict(mynode.left,test_inst)
    else:
        if isinstance(mynode.right,Leaf) :
            return mynode.right.label
        else:
            return predict(mynode.right,test_inst)


# Implement your decision tree below
class DecisionTree():
    tree = {}

    def learn(self, training_set):
    		
        # implement this function
        self.tree = build_tree(training_set)

    # implement this function
    def classify(self, test_instance):
        result = predict(self.tree,test_instance)
        return result

    
def run_decision_tree():

    # Load data set
    # data = np.genfromtxt("test2.txt")
    with open("wine-dataset.csv") as f:
       next(f, None)
       data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    
    print( "Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    accuracy = 0.0
    for j in range(1,11):
        training_set=[x for i,x in enumerate(data) if (i+j)%K != 9]
        test_set = [x for i,x in enumerate(data) if (i+j)%K == 9]

        print ("in to the tree")
        tree = DecisionTree()
    # Construct a tree using training set
        print ("tree is build")
        tree.learn( training_set )
        print (" training is complete")

    # Classify the test set using the tree we just constructed
        results = []
        for instance in test_set:
            result = tree.classify( instance[:-1] )
            results.append( result == instance[-1])

    # Accuracy
        accuracy += float(results.count(True))/float(len(results))
    accuracy = accuracy/10.0
    print ("accuracy: %.4f" % accuracy )      
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()
    
    class Leaf:
        def __init__ (self,label,data):
            self.data = data
            self.label= label
    
    class Node:
        def __init__(self,attr,attr_value,data,rt,lt,right,left):
            self.attr = attr
            self.attr_value = attr_value
            self.data = data
            self.rt = rt
            self.lt = lt
            self.right = right
            self.left = left

if __name__ == "__main__":
    class Leaf:
        def __init__ (self,label,data):
            self.data = data
            self.label= label
    
    class Node:
        def __init__(self,attr,attr_value,data,rt,lt,right,left):
            self.attr = attr
            self.attr_value = attr_value
            self.data = data
            self.rt = rt
            self.lt = lt
            self.right = right
            self.left = left
    run_decision_tree()
