## Name: Bhavna Ramkumar 
## SSID: 862343969CS170_Large_Data__48.txt
## Assigned Small data 25 and Big data 48
## CS170_Large_Data__48.txt
## CS170_Small_Data__25.txt

#import np for faster processing
# pip install numpy
#check np version
# python -c "import numpy; print(numpy.__version__)"

import math 
import numpy as np

# Euclidean Distance Calculation using NumPy for faster proccesing for large datasets 
#np.linalg.norm() another way
# https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
# Refrenced the briefing video also

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# This function performs Leave-One-Out Cross-Validation with the nearest neighbor apporach 
# using numpy for faster calculation 
# # features: input variables 
# # target: target values classes
# #selected: the selected features used for classfication 
# This function performs Leave-One-Out Cross-Validation with the nearest neighbor apporach 
# using numpy for faster calculation 
def leave_one_out(features, target, selected):

# # features: input variables 
# # target: target valies
# #selected: the selected features used for classfication 

    if not selected:
        return 50.0  # If no features are selected, random guessing

    # Convert to NumPy array
    features = np.array(features)  
    target = np.array(target)
    #counter to track correct predictions and total num of instances
    correct = 0
    num = len(features)

    for i in range(num):
        #selected features for test instance
        test_inst = features[i, selected] 
        # removing test from data 
        train_inst = np.delete(features[:, selected], i, axis=0)  
        #delete 
        train_label = np.delete(target, i) 

        # Compute distances bestwweb train_inst- test_inst
        distances = np.linalg.norm(train_inst - test_inst, axis=1)

        #find index of the nearest neighbor 
        #  prediction
        nearest_num = np.argmin(distances)
        predicted = train_label[nearest_num]
        #check if predicted matches the target 
        if predicted == target[i]:
            correct += 1

# Return accuracy percentage
    return (correct / num) * 100  

#Function performs BACKWARD ELIMINATION similar to forward but backwards
def backward_elimination(feat, y):

    # total num of features
    total = len(feat[0])  
    # selected features
    selected = list(range(total)) 
    #call the leave one out function at the start 
    best_accuracy = leave_one_out(feat, y, selected) 
    # this is to store the best feature to be printed at the end
    best_feature_set = selected.copy()

    #print statements 
    print(f"\nRunning nearest neighbor with all {total} features, using 'leave-one-out' evaluation.")
    print(f"I get an accuracy of {best_accuracy:.1f}%\n")
    print("Beginning search.\n")

    # iterate as long as length greater than one 
    while len(selected) > 1:  

        worst_feature = None
        worst_accuracy = 0

        # loop to determine which one to remove 
        #copy the current set, remove feature to test, check accuracy with leave_one_out 
        for feature in selected:
            temp_features = selected.copy()
            temp_features.remove(feature)  
            accuracy = leave_one_out(feat, y, temp_features)
            print(f"    Using feature(s) {set(f + 1 for f in temp_features)}, accuracy is {accuracy:.1f}%")

            if accuracy > worst_accuracy:
                worst_accuracy = accuracy
                worst_feature = feature

        # if a feature is removed and it betters the accuracy than update 

        if worst_feature is not None and worst_accuracy >= best_accuracy:
            selected.remove(worst_feature)
            best_accuracy = worst_accuracy
            best_feature_set = selected.copy()
            print(f"\nFeature set {set(f + 1 for f in selected)} was best, accuracy is {best_accuracy:.1f}%\n")
        else:
            break  

    print(f"\nFinished search!! The best feature subset is {set(f + 1 for f in best_feature_set)}, which has an accuracy of {best_accuracy:.1f}%.")

# function performs FORWARD SELECTION
def forward_selection(feat, y):

    # total num of features 
    total = len(feat[0])  
    #selected features
    selected = []
    #counter for overall accuracy
    overall_accuracy = 0
    #best features
    feature_set = []

    print(f"\nRunning nearest neighbor with all {total} features, using 'leaving-one-out' evaluation.")
    #accuracy of all features calling the other function and printing
    full_accuracy = leave_one_out(feat, y, list(range(total)))
    print(f"I get an accuracy of {full_accuracy:.1f}%\n")
    print("Beginning search.\n")

#add best feature to selected set through looping 
    for inst in range(total):
        #variables to store as loop iterates
        best_feature = None
        best_accuracy = 0

        for feature in range(total):
            #only features that havent been already choosen
            if feature not in selected:
                
                temp_features = selected + [feature]
                #back to function to determine accuracy
                accuracy = leave_one_out(feat, y, temp_features)
                print(f"    Using feature(s) {set(f + 1 for f in temp_features)}, accuracy is {accuracy:.1f}%")

                # check is current accuracy is better than best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature

        # if found add it to selected 
        if best_feature is not None:
            selected.append(best_feature)

            print(f"\nFeature set {set(f + 1 for f in selected)} was best, accuracy is {best_accuracy:.1f}%\n")
            # update 
            if best_accuracy >  overall_accuracy:
                overall_accuracy  = best_accuracy
                feature_set = selected.copy()

        else:
            break  # Stop if no feature improves accuracy

    print(f"\nFinished search!! The best feature subset is {set(f + 1 for f in feature_set)}, which has an accuracy of {overall_accuracy:.1f}%.")



## This function loads the data from the given file  
def load_data(filename):

    #opens file and reads all the lines
    with open(filename, "r") as file:
        lines = file.readlines()  
    data = []

    #converting each line to a list of floats
    for line in lines:
        values = list(map(float, line.split())) 
        data.append(values)

    # getting target values y from the first column 
    y = [row[0] for row in data]   
    # getting feature value x from the rest of the column
    x = [row[1:] for row in data]  

    # prints out the instances and the features for the choosen dataset 
    print(f"This dataset has {len(x[0])} features(not including the class attribute), with {len(x)} instances")
    return x, y


## MAIN FUNCTION: asks for user input for file and the type of search
def main():
    print("Welcome to Bhavna Ramkumars Feature Search Algorithm  ")

    #Enter file name 
    filename = input("Type in the name of the file to test:")

    #Choose 1 for forward algorithm and Choose 2 for Backward Elimination
    print("Type the number of the algorithm you want to run:")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = int(input(""))

    #Loads the data based filename inputted
    x,y = load_data(filename)

    #Goes to the designated functions 
    if choice == 1:
        forward_selection(x,y)
    elif choice == 2:
        backward_elimination(x,y)
    else:
        print("\nInvalid choice. Please enter 1 or 2: ")

main() 



## Previous code worked on using math rather than Numpy 
## Slower proccesing time for largedata sets harded to get results 


# # the equation for  euclidean_distance
# def euclidean_distance(a, b):
#     return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# # This function performs Leave-One-Out Cross-Validation with the neairest neighbor apporach 
# def leave_one_out(features, target, selected):

# # features: input variables 
# # target: target valies
# #selected: the selected features used for classfication

#     #random guessing if not selected
#     if not selected:
#         return 50.0  

#     # counter for correct predication
#     correct = 0
#     # total instances in the dataset using len
#     num = len(features)

#     # iterate through each instance
#     for i in range(num):
#         # get selected features to test out
#         test = [features[i][f] for f in selected]
#         #initalize min distance to infinity 
#         min_distance = float("inf")
#         predicted = None

#         #for loop with nested if statements
#         for j in range(num):
#             #leave one out
#             if i != j:  
#                 # get the selected features and compute the euclidien distance
#                 train_instance = [features[j][f] for f in selected]
#                 distance = euclidean_distance(test, train_instance)

#                 #updating nearest neighbor
#                 if distance < min_distance:
#                     min_distance = distance
#                     predicted = target[j]

#         # increment if predicted is correct 
#         if predicted == target[i]:
#             correct += 1

#     #changing into percentage 
#     accuracy = (correct / num) * 100
#     return accuracy

