import argparse
import train
import nnFunctions
import json


def arg_parser():
    
    parser = argparse.ArgumentParser(description= "Settings for Neuronal Network Prediction")
    
    parser.add_argument('--image_path', dest="image_path", type = str, action="store", default="./flowers/test/46/image_00999.jpg", help = 'Path to image for prediction')
    parser.add_argument('--top_k', dest="top_k", type = int, action="store", default="5", help = 'Top Probabilites displayed')
    parser.add_argument('--arch', dest="arch", type = str, action="store", default="vgg16", help = 'Value for the architecture vgg16 and densenet121')
    parser.add_argument('--category_name', dest="category_name", type = str, action="store", default="cat_to_name.json", help = 'file with Name mapping')
    parser.add_argument('--checkpoint', dest="checkpoint", type = str, action="store", default="checkpoint.pth", help = 'Checkpoint File')
    parser.add_argument('--gpu', dest="gpu", action = "store", default="gpu", help = 'Use GPU')
    parser.add_argument('--data_dir', dest="data_dir", type = str, action="store", default="./flowers/", help = 'Directory from where your data comes from')
    
    args = parser.parse_args()
    return args
    
def main():
    
    args = arg_parser()
    
    data_dir = args.data_dir
    image_path = args.image_path
    arch = args.arch
    top_k = args.top_k
    category_name = args.category_name
    checkpoint = args.checkpoint
    gpu = args.gpu 
    
   
    train_data, train_loader, validation_loader, test_loader, name_of_classes = nnFunctions.load_data(data_dir)
    #Load Checkpoint
    model = nnFunctions.load_checkpoint(checkpoint, arch)
    
    #nnFunctions.imshow(nnFunctions.process_image(image_path))
    
    probability, classes = nnFunctions.predict(image_path, model, top_k)
    
    print(probability)
    print(classes)
    print(name_of_classes)
    with open(category_name, 'r') as f:
        category_name = json.load(f)
        
    flower = [category_name[name_of_classes[e]] for e in classes]
    print()
    
    i = 0          
    while i < top_k:
        print("The images is a {} with probability of {:.4f} %".format(flower[i], probability[i]*100))
        i += 1

    
    
        
if __name__ == '__main__':main()