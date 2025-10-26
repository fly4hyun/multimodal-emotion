def precision(cm, i_, num_classes):
    total = 0
    for num_i in range(num_classes):
        total = total + cm[num_i][i_]
    if total == 0:
        return 0
    return cm[i_][i_] / total
    
def recall(cm, i_, num_classes):
    total = 0
    for num_i in range(num_classes):
        total = total + cm[i_][num_i]
    if total == 0:
        return 0
    return cm[i_][i_] / total

def F1score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def print_result(TrVTe, epoch, correct, total, loss, num_classes, CM, label_mapping):
    if TrVTe == 'train':
        print('epoch : {:3d}, train acc : {:3.3f}% ({:6d} / {:6d}), train loss : {:3.4f}'.format(epoch + 1, 
                                                                                                correct / total * 100, 
                                                                                                correct, 
                                                                                                total, 
                                                                                                loss / total))
    elif TrVTe == 'valid':
        print('epoch : {:3d}, valid acc : {:3.3f}% ({:6d} / {:6d}), valid loss : {:3.4f}'.format(epoch + 1, 
                                                                                                correct / total * 100, 
                                                                                                correct, 
                                                                                                total, 
                                                                                                loss / total))
    else:
        print('epoch : {:3d}, test  acc : {:3.3f}% ({:6d} / {:6d}), test  loss : {:3.4f}'.format(epoch + 1, 
                                                                                                correct / total * 100, 
                                                                                                correct, 
                                                                                                total, 
                                                                                                loss / total))

    total_nums = 0
    F1 = 0
    for i in range(num_classes):
        sum_sample = 0
        print(list(label_mapping.keys())[i], ': [', end = ' ')
        for j in range(num_classes):
            sum_sample = sum_sample + CM[i][j]
            print('{:8d}'.format(CM[i][j]), end = ' ')

        prec = precision(CM, i, num_classes)
        reca = recall(CM, i, num_classes)
        
        F1 = F1 + F1score(prec, reca)
        total_nums = total_nums + sum_sample
        if i == num_classes - 1:
            print('] => {:8d}  {:3.3f}%  --> Macro F1 = {:3.3f}%'.format(sum_sample, reca * 100, F1 / num_classes * 100))
        else:
            print('] => {:8d}  {:3.3f}%'.format(sum_sample, reca * 100))

    return F1 / num_classes * 100
