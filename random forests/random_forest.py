# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import math
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    class_values =[1,0]
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
    #print (p)
    if p==0 or p==1:
        return 0.0
    entropy_value = -(p*math.log(p,2) + (1-p)*math.log((1-p),2))
    #print("entropy")
    #print(entropy_value)
    return entropy_value
    

def test_split(data,value,col) :
    right = list()
    left = list()
    for row in data:
        if row[col] >value:
            right.append(row)
        else :
            left.append(row)
    #print (len(right))
    #print (len(left))
    return right,left    

#def list_attr(data):
 #   n_column = len(data[0])-1
  #  k = int(math.sqrt(n_column))
   # attr_list = random.sample(range(0,n_column),k)
    #print(attr_list)
    #return attr_list
    
def list_attr(data): 
   res = []
   num = int(math.sqrt(len((data[0]))))
   for j in range(num): 
       res.append(random.randint(0, len(data[0])-2)) 
   return res 
    
    
def best_split(data):
    class_values = list(set(row[-1] for row in data))
    #print class_values
    #print (data[0])
    info_gain,attr,attr_value = 0.0,0,0.0
    listofattr = list_attr(data)
    for i in range(len(listofattr)) :
       # print"total length"
       # print len(data[0])-1
        
       #     print i
        unique_set = list((row[listofattr[i]] for row in data))
        unique_set = list(set(unique_set))
     
        for j  in range(len(unique_set)):
            rt,lt = test_split(data,unique_set[j],listofattr[i])
            p = len(rt)/float(len(rt)+len(lt))
            #print("entropy of data")
            #print(entropy(data))
            #print("entropy of lt")
            #print(entropy(lt))
            #print("entropy of rt")
            #print(entropy(rt))
            check_value = entropy(data)-((1-p)*entropy(lt)+p*entropy(rt)) 
           # print ("checkvalue")
            #print (check_value)
            #print ("infogain")
            #print (info_gain)
            if check_value > info_gain:
                #print" kshdfsdgfdsbfdskjhdsfdshfjdsfidsfdsfjadsfsjbf"
                info_gain = check_value
                attr = listofattr[i];
                attr_value = unique_set[j]
        #print "attr"
        #print attr
        #print "attr_value"
        #print attr_value      
    return attr,attr_value,info_gain


                       

def build_tree(data):
    #print (len(data))
    #print (data[0])
    class_values = [1,0]
    
    
    values = list(row[-1] for row in data)
    attr,attr_value,info_gain = best_split(data)
    #print (values)
    #print (entropy(data))
    if info_gain <=0.0 :
        p = values.count(class_values[0])
        if p > (len(data) -p):
            end = Leaf(class_values[0],data)
            return end
        else:
            end = Leaf(class_values[1],data)
            return end
    
    #print ("attr:")
    #print (attr)
    #print ("attr_value:")
    #print (attr_value)
    rt,lt = test_split(data,attr_value,attr)
    #print (len(rt))
    #print (len(lt))
    right = build_tree(rt)
    #print ("right")
    left = build_tree(lt)
    #print ("left")
    mynode = Node(attr,attr_value,data,rt,lt,right,left) 
    #print ("mynode")  
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
    data = np.genfromtxt("test2.txt")
    for i in range(len(data)):
        data[i] = list(data[i])
    data = list(data)
    #print(data)
    
    print ("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    accuracy = 0.0
    
    #training_set=[x for i,x in enumerate(data) if (i+j)%K != 9]
    #test_set = [x for i,x in enumerate(data) if (i+j)%K == 9]
    features_train, test_set = train_test_split(data, test_size = 0.30, random_state = 10)
  #  random_forest = list()
    results = []
    for i in range(10):
        training_set = random.choices(features_train,k=4000)
        print ("in to the tree")
        tree = DecisionTree()
        #random_forest.append(tree)
    # Construct a tree using training set
        print ("tree is build")
        tree.learn( training_set )
        print (" training is complete")

    # Classify the test set using the tree we just constructed
        result1 = []
        for instance in test_set:
            result = tree.classify( instance[:-1] )
            result1.append(result)
        results.append(result1)
    final_result=[]
    counters=0.0
    #print(len(test_set[0]))
    for j in range(len(test_set)):
        t,f = 0,0
        for i in range(10):
            if results[i][j] == 1:
                t=t+1;
            else:
                f=f+1;
        if t>f :
            final_result.append('1')
            if test_set[j][-1] == 1:
                counters = counters+1
        else:
            final_result.append('0')
            if test_set[j][-1] == 0:
                counters = counters+1
        
            
# Accuracy
    
    accuracy = float(counters)/float(len(test_set))
    print ("accuracy of my rf: %.4f" % accuracy )      
    
    x =[]
    y=[]
    for i in range(len(features_train)):
        row=[]
        for j in range(len(features_train[0])-1):
            row.append(features_train[i][j])
        x.append(row)
        y.append(features_train[i][-1])
      
    x_test =[]
    y_test = []  
    for i in range(len(test_set)):
        row=[]
        for j in range(len(test_set[0])-1):
            row.append(test_set[i][j])
        x_test.append(row)
        y_test.append(test_set[i][-1])
    clf = RandomForestClassifier(n_estimators=10,criterion = 'entropy',max_features ='auto')                           
    clf.fit(x, y)
    y_pred_train_nb = clf.predict(x_test)
    acc = accuracy_score(y_test,y_pred_train_nb , normalize = True)
    print (" test accuracy of sklearn rf: %.10f" % acc)
    
    clf = RandomForestClassifier(n_estimators=50,criterion = 'entropy',max_features ='auto',oob_score='True')
    clf.fit(x, y)
    ooberror = 1.0 - (clf.oob_score_)
    print (" out of bag error: %.10f" % ooberror)
    
    
    
    

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
