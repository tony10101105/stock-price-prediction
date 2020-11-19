import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import models
import DataSet


def load_checkpoint(gru_path, cnn_path, mix_path):
    
    GRU = models.numericalRegression()
    CNN = models.textualRegression()
    MIX = models.mixRegression()

    gru_checkpoint = torch.load(gru_path)
    cnn_checkpoint = torch.load(cnn_path)
    mix_checkpoint = torch.load(mix_path)

    gru_optimizer = torch.optim.Adam(GRU.parameters(), lr = args.lr, weight_decay = 1e-3)
    cnn_optimizer = torch.optim.Adam(CNN.parameters(), lr = args.lr, weight_decay = 1e-3)
    mix_optimizer = troch.optim.Adam(mix.parameters(), lr = args.lr, weight_decay = 1e-3)

    gru_optimizer.load_state_dict(gru_checkpoint['optimizer_state_dict'])
    cnn_optimizer.load_state_dict(cnn_checkpoint['optimizer_state_dict'])
    mix_optimizer.load_state_dict(mix_checkpoint['optimizer_state_dict'])

    GRU.load_state_dict(gru_checkpoint['model_state_dict'])
    CNN.load_state_dict(cnn_checkpoint['model_state_dict'])
    MIX.load_state_dict(mix_checkpoint['model_state_dict'])

    assert gru_checkpoint['epoch'] == cnn_checkpoint['epoch'] == mix_checkpoint['epoch'], 'epoch number loading error'
    current_epoch = gru_checkpoint['epoch']

    return GRU, CNN, MIX, gru_optimizer, cnn_optimizer, mix_optimizer, current_epoch

def get_mix_params(MIX):
    param_list = []
    for param in MIX.parameters():
        param_list.append(param.data)
    return param_list

def get_MCC(TP, FP, FN, TN):
    MCC = (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(1/2)
    return MCC
def get_accuracy(TP, FP, FN, TN):
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    return accuracy



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--lr", type = float, default = 1e-2)
parser.add_argument("--n_epochs", type = int, default = 3)
parser.add_argument("--mode", type = str, default = 'train')
parser.add_argument("--GPU", type = bool, default = False)
parser.add_argument("--clipping_value", type = float, default = 1)
parser.add_argument("--sliding_windows", type = list, default = [20, 40, 60])
args = parser.parse_args()
print(args)


print('loading the {} dataset...'.format(args.mode))
Data = DataSet.DataSet(sliding_window = args.sliding_windows, mode = args.mode)
dataLoader = DataLoader(dataset = Data, batch_size = args.batch_size, shuffle = True)
print('datasets loading finished!')

current_epoch = 0

if os.path.exists('./checkpoints/gru_checkpoint.pth') and os.path.exists('./checkpoints/cnn_checkpoint.pth') and os.path.exists('./checkpoints/mix_checkpoints.pth'):
    gru_path = './checkpoints/gru_checkpoint.pth'
    cnn_path = './checkpoints/cnn_checkpoint.pth'
    mix_path = './checkpoints/mix_checkpoint.pth'
    
    GRU, CNN, MIX, gru_optimizer, cnn_optimizer, mix_optimizer, current_epoch = load_checkpoint(gru_path, cnn_path, mix_path)
    
else:
    GRU = models.numericalRegression()
    CNN = models.textualRegression()
    MIX = models.mixRegression()

    gru_optimizer = torch.optim.Adam(GRU.parameters(), lr = args.lr, weight_decay = 1e-3)
    cnn_optimizer = torch.optim.Adam(CNN.parameters(), lr = args.lr, weight_decay = 1e-3)
    mix_optimizer = torch.optim.Adam(MIX.parameters(), lr = args.lr, weight_decay = 1e-3)

if args.mode == 'train':
    GRU.train()
    CNN.train()
    MIX.train()
elif args.mode == 'test':
    GRU.eval()
    CNN.eval()
    MIX.eval()

if args.GPU == True and torch.cuda.is_available():
    print('using GPU...')
    GRU = GRU.cuda()
    CNN = CNN.cuda()
    MIX = MIX.cuda()


#setting the loss function
criterionBCE = nn.BCELoss()

if current_epoch >= args.n_epochs:
    raise Exception('epoch number error!')

#varaibles for calculating losses and the accuracy/MCC
mix_loss = 0
cnn_loss = 0
gru_loss = 0
TP, FP, FN, TN = 0, 0, 0, 0#4 elements of confusion metrix for calculating MCC 

#start training / testing

if args.mode == 'train':
    print('start running on train mode...')
    for epoch in range(current_epoch, args.n_epochs):
        print('epoch:', epoch + 1)
        for i, (price, date, news_vec, nextPrice) in enumerate(dataLoader):

            if nextPrice[0] > price[0][-1]:
                nextPrice[0] = 1
            else:
                nextPrice[0] = 0
        
            price = price.unsqueeze(1)
            news_vec = news_vec.unsqueeze(1)
            price = price.transpose(1,2)
            nextPrice = nextPrice.unsqueeze(1)

            if args.GPU == True and torch.cuda.is_available():
                price = price.cuda()
                news_vec = news_vec.cuda()
                nextPrice = nextPrice.cuda()

            ##train MIX
            
            numericalPred = GRU(price)
            textualPred = CNN(news_vec)
            mixInput = torch.cat((numericalPred, textualPred), 1)
            if args.GPU == True and torch.cuda.is_available():
                mixInput = mixInput.cuda()
            
            mixPred = MIX(mixInput)
            mixLoss = criterionBCE(mixPred, nextPrice)
            mix_optimizer.zero_grad()
            mixLoss.backward()
        
            nn.utils.clip_grad_norm_(MIX.parameters(), args.clipping_value)
            nn.utils.clip_grad_norm_(GRU.parameters(), args.clipping_value)
            nn.utils.clip_grad_norm_(CNN.parameters(), args.clipping_value)
        
            mix_optimizer.step()
            '''print('a:', MIX.linear.weight.data)
            print('a:', CNN.linear[0].weight.grad)
            print('a:', CNN.conv5[2].weight.grad)
            print('a:', CNN.conv5[0].weight.grad)
            print('a:', CNN.conv4[2].weight.grad)
            print('a:', CNN.conv4[0].weight.grad)
            print('a:', CNN.conv3[2].weight.data)
            print('a:', CNN.conv3[0].weight.data)
            print('a:', CNN.conv2[2].weight.grad)
            print('a:', CNN.conv2[0].weight.grad)
            print('a:', CNN.conv1[2].weight.grad)
            print('a:', CNN.conv1[0].weight.grad)'''


            #recording the loss of MIX and updating the confusion metrix
            if (mixPred > 0.5 and nextPrice == 1):
                TP += 1
            elif (mixPred < 0.5 and nextPrice == 0):
                TN += 1
            elif (mixPred > 0.5 and nextPrice == 0):
                FP += 1
            elif (mixPred < 0.5 and nextPrice == 1):
                FN += 1

            mix_loss += torch.sum(mixLoss)
        
            #train GRU

            numericalPred = GRU(price)
            textualPred = CNN(news_vec.detach())   
            mix_params = get_mix_params(MIX)#mix_params[0] are two weights, mix_params[1] is a bias
        
            GRUanswer = Variable((nextPrice - mix_params[1] - textualPred * mix_params[0][0][1]) / mix_params[0][0][0])
            GRUanswer.requires_grad = False
        
            gruLoss = criterionBCE(torch.sigmoid(numericalPred), torch.sigmoid(GRUanswer))
            gru_optimizer.zero_grad()
            gruLoss.backward()
            nn.utils.clip_grad_norm_(GRU.parameters(), args.clipping_value)
            gru_optimizer.step()

            #recording the loss of GRU
            gru_loss += torch.sum(gruLoss)

            ##train CNN
        
            numericalPred = GRU(price.detach())
            textualPred = CNN(news_vec)   
            mix_params = get_mix_params(MIX)#mix_params[0] are two weights, mix_params[1] is a bias
        
            CNNanswer = Variable((nextPrice - mix_params[1] - numericalPred * mix_params[0][0][0]) / mix_params[0][0][1])
            CNNanswer.requires_grad = False
        
            cnnLoss = criterionBCE(torch.sigmoid(textualPred), torch.sigmoid(CNNanswer))
            cnn_optimizer.zero_grad()
            cnnLoss.backward()
            nn.utils.clip_grad_norm_(CNN.parameters(), args.clipping_value)
            cnn_optimizer.step()

            #recording the loss of CNN
            cnn_loss += torch.sum(cnnLoss)


            if (i + 1) % 500 == 0:
                accuracy = get_accuracy(TP = TP, FP = FP, FN = FN, TN = TN)
                MCC = get_MCC(TP = TP, FP = FP, FN = FN, TN = TN)
                print("iteration: {} / {}, Epoch: {} / {}, mix_loss: {:.4f}, gru_loss: {:.4f}, cnn_loss: {:.4f}, accuracy: {:.4f}, MCC: {:.4f}".format(
                    str((i+1)*args.batch_size), str(len(data)), epoch+1, args.n_epochs, mix_loss.data / (500*args.batch_size), gru_loss.data / (500*args.batch_size), cnn_loss.data / (500*args.batch_size), accuracy, MCC))
                mix_loss = 0
                gru_loss = 0
                cnn_loss = 0
                TP, FP, FN, TN = 0, 0, 0, 0


        torch.save({'epoch': epoch+1, 'model_state_dict': GRU.state_dict(), 'optimizer_state_dict': gru_optimizer.state_dict()}, './checkpoints/gru_checkpoint.pth')
        torch.save({'epoch': epoch+1, 'model_state_dict': CNN.state_dict(), 'optimizer_state_dict': cnn_optimizer.state_dict()}, './checkpoints/cnn_checkpoint.pth')
        torch.save({'epoch': epoch+1, 'model_state_dict': MIX.state_dict(), 'optimizer_state_dict': mix_optimizer.state_dict()}, './checkpoints/mix_checkpoint.pth')


elif args.mode == 'test':
    print('start running on test mode...')
    print('testing data length: ', len(dataLoader))
    for i, (price, date, news_vec, nextPrice) in enumerate(dataLoader):
        
        if nextPrice[0] > price[0][-1]:
            nextPrice[0] = 1
        else:
            nextPrice[0] = 0
        
        price = price.unsqueeze(1)
        news_vec = news_vec.unsqueeze(1)
        price = price.transpose(1,2)
        nextPrice = nextPrice.unsqueeze(1)

        if args.GPU == True and torch.cuda.is_available():
            price = price.cuda()
            date = date.cuda()
            news_vec = news_vec.cuda()
            nextPrice = nextPrice.cuda()

        ##test MIX
            
        numericalPred = GRU(price)
        textualPred = CNN(news_vec)
        mixInput = torch.cat((numericalPred, textualPred), 1)
        if args.GPU == True and torch.cuda.is_available():
            mixInput = mixInput.cuda()
            
        mixPred = MIX(mixInput)
        mixLoss = criterionBCE(mixPred, nextPrice)

        #recording the loss of MIX and updating the confusion metrix
        if (mixPred > 0.5 and nextPrice == 1):
            TP += 1
        elif (mixPred < 0.5 and nextPrice == 0):
            TN += 1
        elif (mixPred > 0.5 and nextPrice == 0):
            FP += 1
        elif (mixPred < 0.5 and nextPrice == 1):
            FN += 1
        mix_loss += torch.sum(mixLoss)
        
        #test GRU

        numericalPred = GRU(price)
        textualPred = CNN(news_vec)   
        mix_params = get_mix_params(MIX)#mix_params[0] are two weights, mix_params[1] is a bias
        GRUanswer = Variable((nextPrice - mix_params[1] - textualPred * mix_params[0][0][1]) / mix_params[0][0][0])
        gruLoss = criterionBCE(torch.sigmoid(numericalPred), torch.sigmoid(GRUanswer))

        #recording the loss of GRU
        gru_loss += torch.sum(gruLoss)

        ##test CNN
        
        numericalPred = GRU(price)
        textualPred = CNN(news_vec)   
        mix_params = get_mix_params(MIX)#mix_params[0] are two weights, mix_params[1] is a bias
        CNNanswer = Variable((nextPrice - mix_params[1] - numericalPred * mix_params[0][0][0]) / mix_params[0][0][1])
        cnnLoss = criterionBCE(torch.sigmoid(textualPred), torch.sigmoid(CNNanswer))

        #recording the loss of CNN
        cnn_loss += torch.sum(cnnLoss)

    MCC = get_MCC(TP = TP, FP = FP, FN = FN, TN = TN)
    print('mix_loss: {:.4f}, gru_loss: {:.4f}, cnn_loss: {:.4f}, testing accuracy: {:.4f}, MCC: {:.4f}'.format(mix_loss / len(dataLoader), gru_loss / len(dataLoader), cnn_loss / len(dataLoader), mix_correct_num / len(dataLoader), MCC))
    mix_loss = 0
    gru_loss = 0
    cnn_loss = 0
    TP, FP, FN, TN = 0, 0, 0, 0

    
        
    
        
        





        
