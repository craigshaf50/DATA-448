import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def cost_function(y_true, y_pred):
    
    '''
    This a customize scoring function that takes two arguments:
    y_true: true labels
    y_pred: likelihoods from the model   
    '''
    
    ## Defining cutoff values in a data-frame
    results = pd.DataFrame({'cutoffs': np.round(np.linspace(0.05, 0.95, num = 40, endpoint = True), 2)})
    results['cost'] = np.nan
    
    for i in range(0, results.shape[0]):
        
        #changing likelihoods to labels
        y_pred_lab = np.where(y_pred < results['cutoffs'][i], 0, 1)
        
        #computing confusion matrix and scoring based on description
        x = confusion_matrix(y_pred_lab, y_true)
        results['cost'][i] = -25 * x[1, 0] - 5 * x[0, 1] + 5 * x[1, 1]
        
    #sorting results 
    results = results.sort_values(by = 'cost', ascending = False).reset_index(drop = True)
    
    return results['cost'][0]


def cost_function_cutoff(y_true, y_pred):
    
    #defining cutoff values in a data-frame
    results = pd.DataFrame({'cutoffs': np.round(np.linspace(0.05, 0.95, num = 40, endpoint = True), 2)})
    results['cost'] = np.nan
    
    for i in range(0, results.shape[0]):
        
        #changing likelihoods to labels
        y_pred_lab = np.where(y_pred < results['cutoffs'][i], 0, 1)
        
        #computing confusion matrix and scoring based on description
        x = confusion_matrix(y_pred_lab, y_true)
        results['cost'][i] = -25 * x[1, 0] - 5 * x[0, 1] + 5 * x[1, 1]
        
    #sorting results 
    results = results.sort_values(by = 'cost', ascending = False).reset_index(drop = True)
    print("Score: ", results['cutoffs'])   
    
    return results['cutoffs'][0]