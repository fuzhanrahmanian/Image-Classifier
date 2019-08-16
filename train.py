import argparse
import nnFunctions



def arg_parser():
    
    parser = argparse.ArgumentParser(description= "Settings for Neuronal Network Traning")
    
    parser.add_argument('--arch', dest="arch", type = str, action="store", default="vgg16", help = 'Value for the architecture vgg16 and densenet121')
    parser.add_argument('--save_dir', dest="save_dir", type = str, action="store", default="./checkpoint.pth", help = 'Directory where to save the trained Network')
    parser.add_argument('--data_dir', dest="data_dir", type = str, action="store", default="./flowers/", help = 'Directory from where your data comes from')
    parser.add_argument('--learning_rate', dest="learning_rate", type = int, action="store", default=0.001, help = 'Learning rate for the NN')
    parser.add_argument('--hidden_layers', dest="hidden_layers", type = int, action="store", default= 4096, help = 'Number of hidden layers of the NN')
    parser.add_argument('--dropout', dest="dropout", type = int, action="store", default= 0.15, help = 'Dropout of the NN')
    parser.add_argument('--epochs', dest="epochs", type = int, action="store", default= 5, help = 'Number of Epochs of the NN')
    parser.add_argument('--gpu', dest="gpu", action = "store", default="gpu", help = 'Use GPU')
    
    args = parser.parse_args()
    return args


def main():
    
    args = arg_parser()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    dropout = args.dropout
    gpu = args.gpu 
    
    print(arch)
    print(epochs)
    #Load the data
    train_data, train_loader, validation_loader, test_loader, name_of_classes = nnFunctions.load_data(data_dir)
    
    #Create the NN
    model, criterion, optimizer, device = nnFunctions.network(arch, hidden_layers, learning_rate, gpu, dropout, name_of_classes)
    
    #Train the CNN
    nnFunctions.train_network(epochs, model, train_loader, optimizer, criterion, device, save_dir, validation_loader, test_loader, train_data)
    
    return arch, device, name_of_classes

if __name__ == '__main__':main()