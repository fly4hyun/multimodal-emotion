import torch

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
                                                                                                loss))
    elif TrVTe == 'valid':
        print('epoch : {:3d}, valid acc : {:3.3f}% ({:6d} / {:6d}), valid loss : {:3.4f}'.format(epoch + 1, 
                                                                                                correct / total * 100, 
                                                                                                correct, 
                                                                                                total, 
                                                                                                loss))
    else:
        print('epoch : {:3d}, test  acc : {:3.3f}% ({:6d} / {:6d}), test  loss : {:3.4f}'.format(epoch + 1, 
                                                                                                correct / total * 100, 
                                                                                                correct, 
                                                                                                total, 
                                                                                                loss))

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

def MeanAcc(correct, total, re_correct, re_total, output, label, num_classes, k_top):
    
    co_output = (output >= output.sort(-1)[0][:, -k_top].unsqueeze(1)) * label
    re_output = (output < output.sort(-1)[0][:, -k_top].unsqueeze(1)) * (1 - label)
    for i in range(num_classes):
        
        correct[i] = correct[i] + co_output.sum(0)[i].item()
        total[i] = total[i] + label.sum(0)[i].item()
        
        re_correct[i] = re_correct[i] + re_output.sum(0)[i].item()
        re_total[i] = re_total[i] + (1 - label).sum(0)[i].item()
    
    return correct, total, re_correct, re_total

def Mean7Acc(correct, total, re_correct, re_total, output, label, num_classes, k_top):
    
    co_output = output * label
    re_output = (1 - output) * (1 - label)
    for i in range(num_classes):
        
        correct[i] = correct[i] + co_output.sum(0)[i].item()
        total[i] = total[i] + label.sum(0)[i].item()
        
        re_correct[i] = re_correct[i] + re_output.sum(0)[i].item()
        re_total[i] = re_total[i] + (1 - label).sum(0)[i].item()
    
    return correct, total, re_correct, re_total

def Result_MeanACC(epoch, correct, total, loss, correct_7, total_7, cor, tot, rcor, rtot, cor_7, tot_7, rcor_7, rtot_7, emotiom_mapping, num_classes, TrVTe):

    acc_all = 0
    for i in range(num_classes):
        if tot[i] == 0:
            tot[i] = 1
        acc_all = acc_all + cor[i] / tot[i] * 100
    acc_all = acc_all / num_classes
    
    if TrVTe == 'train':
        print('epoch : {:3d}, train acc : {:3.3f}% ({:6d} / {:6d}), train loss : {:3.4f}, train mean acc : {:3.3f}%, train 7 acc : {:3.3f}%'.format(epoch + 1, 
                                                                                        correct / total * 100, 
                                                                                        correct, 
                                                                                        total, 
                                                                                        loss, 
                                                                                        acc_all, 
                                                                                        correct_7 / total_7 * 100))

    elif TrVTe == 'valid':
        print('epoch : {:3d}, valid acc : {:3.3f}% ({:6d} / {:6d}), valid loss : {:3.4f}, valid mean acc : {:3.3f}%, valid 7 acc : {:3.3f}%'.format(epoch + 1, 
                                                                                        correct / total * 100, 
                                                                                        correct, 
                                                                                        total, 
                                                                                        loss, 
                                                                                        acc_all, 
                                                                                        correct_7 / total_7 * 100))
        
    elif TrVTe == 'test':
        print('epoch : {:3d}, test  acc : {:3.3f}% ({:6d} / {:6d}), test  loss : {:3.4f}, test  mean acc : {:3.3f}%, test  7 acc : {:3.3f}%'.format(epoch + 1, 
                                                                                        correct / total * 100, 
                                                                                        correct, 
                                                                                        total, 
                                                                                        loss, 
                                                                                        acc_all, 
                                                                                        correct_7 / total_7 * 100))
        
    j = 0
    if tot_7[j] == 0:
        tot_7[j] = 1
    if rtot_7[j] == 0:
        rtot_7[j] = 1    

    print("{:1.4f}".format(cor_7[j] / tot_7[j]), end = ' ')
    print("({:5d} / {:5d}) /".format(int(cor_7[j]), int(tot_7[j])), end = ' ')
    print("{:1.4f}".format(rcor_7[j] / rtot_7[j]), end = ' ')
    print("({:5d} / {:5d})".format(int(rcor_7[j]), int(rtot_7[j])), end = ' ')
    print('  ==> ', end = ' ')
    j = 1
    for i in range(num_classes):
        
        print(list(emotiom_mapping.keys())[i], ':', end = ' ')
        print("{:1.2f}/{:1.2f}".format(cor[i] / tot[i], rcor[i] / rtot[i]), end = ' ')

        if i == 1 or i == 5 or i == 9 or i == 21 or i == 29 or i == 30:
            print()
            if tot_7[j] == 0:
                tot_7[j] = 1
            if rtot_7[j] == 0:
                rtot_7[j] = 1
            #print("{:1.4f}/{:1.4f}".format(cor_7[j] / tot_7[j], rcor_7[j] / rtot_7[j]), end = ' ')
            print("{:1.4f}".format(cor_7[j] / tot_7[j]), end = ' ')
            print("({:5d} / {:5d}) /".format(int(cor_7[j]), int(tot_7[j])), end = ' ')
            print("{:1.4f}".format(rcor_7[j] / rtot_7[j]), end = ' ')
            print("({:5d} / {:5d})".format(int(rcor_7[j]), int(rtot_7[j])), end = ' ')
            print('  ==> ', end = ' ')

            j = j + 1
            
        if i == 33:
            print()

def ktop_result(output, label_34, ktop, mapping_34, label_7, mapping_7, masking):
    
    output_pred = (output >= output.sort(-1)[0][:, -ktop].unsqueeze(1)) * 1
    output_ox = ((output_pred * label_34).sum() > 0.1).item()
    
    ox = 'O' if output_ox else 'X'

    multi_label = ' '.join([mapping_34[i] for i in range(34) if label_34[i] == 1])
    pred = ' '.join([mapping_34[i] for i in range(34) if output_pred.squeeze(0)[i] == 1])

    p = output * (output >= output.sort(-1)[0][:, -ktop].unsqueeze(1))
    p = ' '.join([str(p[0, i].item())[:6] for i in range(34) if p[0, i] != 0])

    uni_temp_list = []
    for i in range(7):
        
        uni_temp = (output >= output.sort(-1)[0][:, -ktop].unsqueeze(1))
        uni_temp_list.append(((uni_temp * (masking == i)).sum(-1) > 0.1).unsqueeze(1) * 1.0)

    train_uni_output = torch.cat(uni_temp_list, dim = 1)
    
    ox_7 = 'O' if (label_7 * train_uni_output).sum() > 0.1 else 'X'
    pred_7 = ' '.join([mapping_7[i] for i in range(7) if train_uni_output.squeeze(0)[i] == 1])

    return ox, multi_label, pred, p, ox_7, pred_7
