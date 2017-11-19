from sklearn.metrics import precision_score,recall_score,average_precision_score

def round_manual(data, threshold):
    return (data >= threshold).astype(int)

def calculate_accuracy(predicted, actuals, num_labels):
    """
    @param predicted: data type = Variable
    @param actuals: data type = Variable
    @param num_labels: no of go terms
    @return: accuracy measure
    """
    predicted = np.round(predicted.data.numpy())
    total_predictions = actuals.size()[0]
    accuracy = np.sum(predicted==actuals.data.numpy())/(total_predictions*num_labels)
    return accuracy


def recall_precision_ProteinMethod(predicted, actual):
    '''
    Overall, this function calculates the recall and precision of the validation set proteins.
    The function FIRST calculates the precision and recall values of INDIVIDUAL proteins. 
    It then takes the mean average of these values to get "dataset-level" precision and recall.
    '''
    
    PositivesPerRow = actual.numpy().sum(axis=1) #number of functions for each protein
    PosPredictionsPerRow = predicted.sum(axis=1) #number of predictions for each protein
    TPs = np.multiply(actual.numpy(), predicted) #element-wise multiplication: 1 if TP, else 0
    TPsPerRow = TPs.sum(axis=1) #number of true positives for each protein
    
    #PrecisionPerRow (Protein) - if protein has 0 positive predictions, the protein's precision = 0.
    #Else, the protein's precision = TPs/PositivePreds
    PrecisionPerRow = np.where(PosPredictionsPerRow == 0, 0, TPsPerRow/PosPredictionsPerRow)
    RecallPerRow = np.where(PositivesPerRow==0, 0, TPsPerRow/PositivesPerRow) #Recall per Protein
    
    #RecallScore = average of individual protein recall scores
    RecallScore = sum(RecallPerRow)/len(RecallPerRow) #denominator is non-zero
    
    #PrecisionScore = average of CERTAIN individual protein precision scores (see line below)
    #Only consider rows with at least one predicted Go-Term.
    #Note that some proteins can have Precision=0 but still have predictions.
    if sum(PrecisionPerRow)>0:
        PrecisionScore = sum(PrecisionPerRow)/len([x for x in PosPredictionsPerRow if x!=0]) 
    else:
        PrecisionScore = 0
    return RecallScore, PrecisionScore

def recall_precision_GoTermMethod(predicted, actual):
    '''
    The function FIRST calculates the precision and recall values of INDIVIDUAL Go-Terms. 
    It then takes the mean average of these values to get "dataset-level" precision and recall.
    '''
    PositivesPerGoTerm = actual.numpy().sum(axis=0) #number of functions for each protein
    PosPredictionsPerGoTerm = predicted.sum(axis=0) #number of predictions for each protein
    TPs = np.multiply(actual.numpy(), predicted) #element-wise multiplication: 1 if TP, else 0
    TPsPerGoTerm = TPs.sum(axis=0) #number of true positives for each protein
    
    PrecisionPerGoTerm = np.where(PosPredictionsPerGoTerm == 0, 0, TPsPerGoTerm/PosPredictionsPerGoTerm)
    RecallPerGoTerm = np.where(PositivesPerGoTerm==0, 0, TPsPerGoTerm/PositivesPerGoTerm) #Recall per Protein
    
    #RecallScore = average of individual Go Term recall scores
    RecallScore = sum(RecallPerGoTerm)/len(RecallPerGoTerm) #denominator is non-zero
    PrecisionScore = sum(PrecisionPerGoTerm)/len(PrecisionPerGoTerm)
    return RecallScore, PrecisionScore
        
def F_score(predicted, actuals, method = 'GoTerm'):
    """
    @param predicted: data type = Variable
    @param actuals: data type = Variable
    @return: Maximum f score over all values of tau and the corresponding tau threshold
    """
    f_max, optimal_threshold, optimal_precision, optimal_recall = 0, 0, 0, 0
    for threshold in [i/100 for i in range(1,100)]:
        predicted_tau = round_manual(predicted.data.numpy(), threshold)
        
        if method == 'GoTerm':
            recall_score, precision_score = recall_precision_GoTermMethod(predicted_tau, actuals)
        elif method == 'Protein':
            recall_score, precision_score = recall_precision_ProteinMethod(predicted_tau, actuals)
        
        if recall_score==0 and precision_score==0:
            output = 0
        else:
            output = ((2*precision_score*recall_score) / (precision_score + recall_score))
        if output > f_max:
            f_max = output
            optimal_threshold = threshold
            optimal_precision = precision_score
            optimal_recall = recall_score
    return f_max, optimal_threshold, optimal_precision, optimal_recall