import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import network

# Training phase
def train(args, model, device, train_loader, optimizer):
    model.train()
    # cross entropy loss
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (image, target) in enumerate(train_loader):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        # Save the model checkpoint
        if args.save_model and epoch % 10 == 0:
            torch.save(model.state_dict(), 'model/basic_iter_model-{}.ckpt'.format(epoch))


#dv_images-args.epsilon, next_adv_images) Test phase
def test(args, model, device, test_loader):
    model.eval()
    
    if args.model_parameter is not None:
        model.load_state_dict(torch.load(args.model_parameter))
    
    correct = 0
    adv_correct = 0
    misclassified = 0
    
    criterion = nn.CrossEntropyLoss()

    # because of image value [0, 1], args.epsilon * 255
    # number of iteration
    num_iter = int(min(args.epsilon*255 +4, 1.25*args.epsilon*255))
    print('number of adversarial example iteration :', num_iter)
    for idx, (images, targets) in enumerate(test_loader):
        images = Variable(images.to(device), requires_grad=True)
        targets = Variable(targets.to(device))
        
        origin_output = model(images)
        
        # X_0 adv images
        adv_images = images
        for i in range(num_iter):
            outputs = model(adv_images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Generate perturbation
            grad_j = torch.sign(adv_images.grad.data)
            next_adv_images = adv_images + args.epsilon * grad_j
            lower_adv_images = torch.max(torch.tensor(0.).to(device),torch.max(adv_images-args.epsilon, next_adv_images)) 
            adv_images = torch.min(torch.tensor(1.).to(device), torch.min(adv_images+args.epsilon, lower_adv_images))
            adv_images = Variable(adv_images.to(device), requires_grad=True) 
        # test adversarial example
        adv_output = model(adv_images)
        
        _, preds = torch.max(origin_output.data, 1)
        _, adv_preds = torch.max(adv_output.data, 1)
        
        correct += (preds == targets).sum().item()
        adv_correct += (adv_preds == targets).sum().item()
        misclassified += (preds != adv_preds).sum().item()
        current_test_size = args.test_batch_size*(idx+1)
        print('\n Epoch {} : correct({}/{}) , adversarial correct({}/{}), misclassify({}/{})\n'.format(
            idx, correct, current_test_size, adv_correct, current_test_size, misclassified, current_test_size
        ))
        
    print('\n Testset Accuracy: {}/ {} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        ))
    print('\n Adversarial Testset Accuracy: {}/ {} ({:.2f}%)\n'.format(
            adv_correct, len(test_loader.dataset),
            100. * adv_correct / len(test_loader.dataset)
        ))
    print('\n misclassified examples : {}/ {}\n'.format(
            misclassified, len(test_loader.dataset)
        ))

        

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--dataset-normalize', action='store_true' , default=False,
                        help='input whether normalize or not (default: False)')
    parser.add_argument('--test-mode', action='store_true', default=False,
                        help='input whether training or not (default: False')
    parser.add_argument('--model-parameter', type=str, default=None,
                        help='if test mode, input model parameter path')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    transformation = transforms.ToTensor()
    # Dataset normalize
    if args.dataset_normalize:
        transformation = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('./data', train=True, download=True,
                        transform=transformation),
      batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('./data', train=False, download=True,
                        transform=transformation),
      batch_size=args.test_batch_size, shuffle=True)
  
    model = network.GoogLeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.test_mode is False:
        train(args, model, device, train_loader, optimizer)
        test(args, model, device, test_loader)
    else:
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()            

