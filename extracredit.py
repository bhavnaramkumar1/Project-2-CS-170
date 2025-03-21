# extra credit 
#importing numpy for faster processig 
#importing pandaspd for csv files
#google sheet with SD and normalized data stores as a csv file 


import numpy as np
import pandas as pd

# Function to load and process CSV file
def load_data(filename):
    # Read CSV file, making sure no space present
    datas = pd.read_csv(filename, skipinitialspace=True)  

    # taking out spaces from colums
    datas.columns = datas.columns.str.strip()
    # coverting to int class
    y = datas.iloc[:, 0].astype(int).values  
    # convert to float features
    x = datas.iloc[:, 1:].astype(float).values  

    print(f"\nThis dataset has {x.shape[1]} features (not including the class attribute), with {x.shape[0]} instances.")
    return x, y

#  calculate Euclidean Distance using NumPy same as orginal code 
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1)) 

#same as orginal code 
def leave_one_out(features, target, selected):

    if not selected:
        return 50.0  # If no features are selected, random guessing

    # Convert to numpy array
    features = np.array(features)  
    target = np.array(target)

    #correct predictions and total num of instances
    correct = 0
    num = len(features)

    for i in range(num):
        #selected features for test instance
        test_inst = features[i, selected] 
        # removing test from data 
        train_inst = np.delete(features[:, selected], i, axis=0)  
        train_label = np.delete(target, i) 

        # Compute distances train_inst- test_inst
        distances = np.linalg.norm(train_inst - test_inst, axis=1)

        #find index of the nearest neighbor and prediction
        nearest_num = np.argmin(distances)
        predicted = train_label[nearest_num]

        #check if predicted matches the target 
        if predicted == target[i]:
            correct += 1

    return (correct / num) * 100  

#same as orginal
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
            break  # Stop if no  improves accuracy
    print(f"\nFinished search!! The best feature subset is {set(f + 1 for f in feature_set)}, which has an accuracy of {overall_accuracy:.1f}%.")




def main():
    print("Welcome to Feature Selection with Nearest Neighbor")
    # file name
    filename = "whitewine.csv"
    # load data
    x, y = load_data(filename)
    # running forward selection same as orginal function
    forward_selection(x, y)

main()
