# test unitaire _ metrics

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def eval_metrics(actual,pred):
    accuracy = round(accuracy_score(actual,pred),2)
    precision= round(precision_score(actual,pred),2)
    recall = round(recall_score(actual,pred),2)
    f1 = round(f1_score(actual,pred),2)
    return accuracy,precision,recall,f1

def test_eval_metrics():
    ac = [0,1,1,0,0,1]  
    pr = [1,1,1,0,1,1]
    pr_accuracy,pr_precision,pr_recall,pr_f1 = eval_metrics(ac,pr)
    true_acc = 0.67
    true_pre = 0.6
    true_rec = 1.0
    true_f1 = 0.75
    assert((pr_accuracy == true_acc) & (pr_precision == true_pre) & (pr_recall == true_rec) & (pr_f1==true_f1))

test_eval_metrics()