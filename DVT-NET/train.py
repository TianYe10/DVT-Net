
import torch
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

def train_model(model, criterion, optimizer, num_epochs, device, image_datasets, dataloaders, save_path):
    since = time.time()

    best_f1 = 0

    train_f1 = []
    val_f1 = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            train_outputs = []
            train_preds = []
            train_trues = []
            output_tensors = []



            for inputs, vsi, tda, labels in tqdm(dataloaders[phase]):
                tda = tda.float()
                tda = tda.unsqueeze(1).to(device)
                labels = labels.unsqueeze(1).to(device)
                vsi = vsi.unsqueeze(1).to(device)
                labels = labels.to(device)
                outputs = model(inputs, vsi, tda)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                preds = (outputs >= 0)   #depending on loss func.
                preds_array = preds.cpu().numpy()  
                labels_array = labels.data.cpu().numpy()
                outputs_array = outputs.detach().cpu().numpy()

                train_outputs.extend(outputs_array) 
                train_preds.extend(preds_array)
                train_trues.extend(labels_array)
                output_tensors.extend(outputs.detach().cpu())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            sklearn_accuracy = accuracy_score(train_trues, train_preds) 
            sklearn_precision = precision_score(train_trues, train_preds, average='macro')
            sklearn_recall = recall_score(train_trues, train_preds, average='macro')
            sklearn_f1 = f1_score(train_trues, train_preds, average='macro')
            
            output_tensors = torch.sigmoid(torch.tensor(output_tensors)).numpy()
            print('{} loss: {:.4f}, torch acc: {:.4f}, sklearn acc: {:.4f}, precision:{:.4f}, recall: {:.4f}, f1_score:{:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc,
                                                        sklearn_accuracy,
                                                        sklearn_precision,
                                                        sklearn_recall,
                                                        sklearn_f1,))

            if phase == 'train':
              train_f1.append(sklearn_f1)
              train_accuracy.append(sklearn_accuracy)


            if phase == 'validation':
              target_names = ['healthy', 'disease']
              print(classification_report(train_trues, train_preds, target_names=target_names))

              val_f1.append(sklearn_f1)
              val_accuracy.append(sklearn_accuracy)

            if phase == 'validation' and sklearn_f1 > best_f1:
                torch.save(model, save_path + '/best-model-resnet101.pt')
                torch.save(model.state_dict(), save_path + '/best-model-parameters.pt')
                save_outputs = output_tensors
                save_labels = train_trues
                save_preds = train_preds
                best_f1 = sklearn_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                print('best_f1:', best_f1)
                print('A new best model saved at epoch {}!'.format(epoch + 1))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))      
    print('Best model saved to:', save_path)      

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_f1, train_accuracy, val_f1, val_accuracy, save_outputs, save_labels, save_preds