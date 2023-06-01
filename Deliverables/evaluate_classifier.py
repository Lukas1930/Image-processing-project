import pickle
from extract_features import extract_features
import cv2
import sys
import numpy as np

def classify(img):
    #Extract features (the same ones that you used for training)
    x = extract_features(img)
    x = x[np.newaxis, :]      
     
    #Load the trained classifier
    classifier = pickle.load(open('logistic_regression_model.pkl', 'rb'))    
    
    #Use it on this example to predict the label AND posterior probability
    pred_label = classifier.predict(x)
    pred_prob = classifier.predict_proba(x)     
     
    #print('predicted label is ', pred_label)
    #print('predicted probability is ', pred_prob)
    return pred_label, pred_prob 
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_classifier.py <image_path>")
    else:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)

        pred_label, pred_prob = classify(image)

        print('predicted label is ', pred_label)
        print('predicted probability is ', pred_prob)