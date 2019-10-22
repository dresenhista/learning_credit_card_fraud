#the functions below perform the stepwise selection using backward method based on feature weight
#this functionality is not created in python (specially for logistic regression) within the statsmodels package
#the function orders the p-values from the highest to the lowest and then remove one at a time refitting the model 
#each time checking if the features change their relevance. 

import statsmodels.api as sm
import pandas as pd
import numpy as np

#to plot stuff
import matplotlib.pyplot as plt

import scikitplot as skplt
import matplotlib.gridspec as gridspec

#model evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


#function to do the stepwise backward selection
def new_stepwise(X, y, alpha):
    
    #creates empty list
    to_remove = ['']
    
    n_old = len(X.columns)
    n_new = 0
    

    #have we stop seeing high p-values?
    while n_old - n_new>0:
        
        n_old = len(X.columns)
        
        #run the model 
        lr=sm.Logit(y, X)
        lr = lr.fit(disp=0)
        lr.pvalues = lr.pvalues.sort_values(ascending=False)
        
        k = 0
        #are we at the last one of the pvalues
        while k < n_old:


            if lr.pvalues.iloc[k]<alpha: 

                k +=1
            else:
                
                print(lr.pvalues.index[k])
                to_remove.append(lr.pvalues.index[k])
                X = X[X.columns.difference(to_remove)]
                break

                
        n_new = len(X.columns)


    return X.columns

def logit_score(model, X, y, threshold=0.95):
    #function to score the logit model (only works for models that support .predict function)
    y_t = model.predict(X)
    y_t = y_t > threshold
    z= abs(y_t-y)
    score = 1- sum(z)/len(z)
    return score


def confusion_matrix_lr(output_path, model_name, trigger, model, X, y_real, method, financed='none'):
    
    """ 
    function to create the confusion matrix and save it in a selected location and return the predicted value using
    the function. this works in any function that has a binary response and accepts .predict on indepentente matrix
    output_path: file location
    model_name = file name to be used when saving it
    trigger: it the threshold chosen to determine fraud or not
    X: predicted matrix used, usually in the testing set
    y_real = the real values to be compared with
    
    output: --a PNG file saved in the output_path
    -- a vector with the predicted values (1 as success 0 as failure)
    
    """
    
    y_pred = model.predict(X)
    y_pred = y_pred>trigger
    y_pred = y_pred*1
    
    
    save_confusionmatrix(output_path, model_name, trigger, y_real, y_pred, method, financed);

    
    return y_pred

def save_confusionmatrix(output_path, model_name, trigger, y_real, y_pred, method, financed= 'none'):
    
    'for the function above this is where the saving image actually happens'
    
    if method=='pred':
        confusion_matrix = skplt.metrics.plot_confusion_matrix(y_real, y_pred);
    else:
        confusion_matrix = plot_confusion_matrix_value_new(y_real, y_pred, financed);
        
    confusion_matrix.figure.savefig(f'{output_path}confusion_matrix_'+model_name+'_'+str(trigger)+'.JPEG')
    confusion_matrix.figure.show
    return confusion_matrix 




def order_correlation_matrix(df):
    ''' calculates the correlation matrix and order the values from high correlated to low'''
    corr_matrix = df.corr().abs() #calculates correlation matrix
    s = corr_matrix.unstack()
    so=pd.DataFrame(s)
    so = so.reset_index()
    so.columns = ['var1', 'var2', 'corr']
    so = so.sort_values(by='corr', ascending=False)
    so = so[so['corr']!=1]
    
    return so



def remove_features(df, corr_value):
    '''
    receives a dataframe with X matrix and returns features from X that needs to be removed based on 
    '''    
    
    n_before = 0
    n_after = 1
    remove = []
    
    while(n_after>n_before):
        n_before = n_after

        so = order_correlation_matrix(df)
        if(so['corr'].iloc[0]>=corr_value):
            remove.append(so['var1'].iloc[0])
            n_after = len(remove)
    return remove


#Optimum value for threshold
def model_threshold_finder(start, interval, y_real, y_prob, modelname):

    '''
    calculates precision and recall for y_real versus y_pred looping through a set of values between 0 and 1
    input - 
    y_real: is the vector os 1s and 0 of the real data.
    y_prob: is the vector of probabilities predicted by the LR model
    intervall: is how many intervalls you want between 0 and 1 to have the precision and recall to be calculated on
    output - 

    a dataframe with the values
    a plot plotting precision versus recall by threshold 
    '''
    
    threshold_list = np.arange(start, 1.000001, interval)
    precision = []
    recall = []
    trigger = []

    
    for threshold in threshold_list:
        y_pred = y_prob>threshold
        y_pred = y_pred*1
        trigger.append(threshold)
        precision.append(precision_score(y_real, y_pred))
        recall.append(recall_score(y_real, y_pred))
        
    dataframe  = pd.DataFrame(data = {'trigger': trigger, 'precision': precision, 'recall':recall})

    #creates the vector of probabilities of 0 and 1s
    y_probas = np.array([1-y_prob, y_prob])
    y_probas = np.transpose(y_probas)
    
    #plots the precision-recall chart
    skplt.metrics.plot_precision_recall(y_real, y_probas, title = 'Precision-Recall Curve %s' %modelname, 
                                        title_fontsize = 20)
    dataframe['evaluator'] = dataframe['precision'] * dataframe['recall']
    
    return dataframe



def draw_precision_recall(interval_range, dataframe):
    

    fig = plt.figure(1)
    plt.gcf().set_size_inches(12, 8) #sets size of the image    
    gridspec.GridSpec(2,2) #creates grids (in this case is 2 by 2)
    
    
    
    #calculates the global max = 
    ymax = max(dataframe['evaluator'])
    xmax = dataframe[dataframe['evaluator'] == ymax]['trigger'].values[0]

    #plot 1
    plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
    plt.xlabel('Threshold')
    plt.ylabel('Function that maximizes accuracy')
    plt.plot(dataframe['trigger'], dataframe['evaluator'],  color = '#00C00F')
    text= "threshold = {:.3f} \nglobal max ={:.3f}".format(xmax, ymax)
    plt.annotate(text, xy=(xmax-0.35, ymax-0.02))
    
    #plot 2
    plt.subplot2grid((2,2), (0,1), colspan=2)
    plt.xlabel('Threshold')
    plt.ylabel('Precision function')
    plt.plot(dataframe['trigger'], dataframe['precision'], color='#0F2F5B')    
    
    #plot 3
    plt.subplot2grid((2,2), (1,1), colspan=2)
    plt.xlabel('Threshold')
    plt.ylabel('Recall function')        
    plt.plot(dataframe['trigger'], dataframe['recall'], color='#5B0F52')

    
    

    

    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    fig.show()
    return xmax