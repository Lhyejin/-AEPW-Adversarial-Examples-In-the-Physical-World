import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from network import GoogLeNet
from adversarial_attack import fgsm, basic_iteration, least_likely_class


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

 
# Test phase
def test(args, model, device, test_loader):
    model.eval()
    
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
    
    correct = 0
    adv_correct = 0
    misclassified = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for idx, (images, targets) in enumerate(test_loader):
        images = Variable(images.to(device), requires_grad=True)
        targets = Variable(targets.to(device))
        
        origin_output = model(images)

        # Generate Perturbation
        # Iteration Least-Likely Class Method
        if args.attack == 'll_class':
            _, ll_targets = torch.min(origin_output.data, 1)
            num_iter, adv_images = least_likely_class(args, model, device, images, ll_targets, criterion)
        # Basic Iteration Method
        elif args.attack == 'basic_iter':
            num_iter, adv_images = basic_iteration(args, model, device, images, targets, criterion)
        else: # FGSM
            adv_images = fgsm(args, model, device, images, targets, criterion)

        # test adversarial example
        adv_output = model(adv_images)
        
        # Prediction 
        _, preds = torch.max(origin_output.data, 1)
        _, adv_preds = torch.max(adv_output.data, 1)
        
        correct += (preds == targets).sum().item()
        adv_correct += (adv_preds == targets).sum().item()
        misclassified += (preds != adv_preds).sum().item()
        current_test_size = args.test_batch_size*(idx+1)

        # if you want to see intermidiate results, uncomment print
        
        print('\n Epoch {} : correct({}/{}) , adversarial correct({}/{}), misclassify({}/{})'.format(
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
    
    if args.attack == 'basic_iter' or args.attack == 'll_class':
        print('Number of adversarial example iteration :', num_iter)
        

def main():
    parser = argparse.ArgumentParser(description='PyTorch White-Box Attack')
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
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='epsilon of adversarial attack')
    parser.add_argument('--dataset-normalize', action='store_true' , default=False,
                        help='input whether normalize or not (default: False)')
    parser.add_argument('--test-mode', action='store_true', default=False,
                        help='input whether training or not (default: False')
    parser.add_argument('--model', type=str, default=None,
                        help='if test mode, input model parameter path')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--attack', type=str, default='fgsm',
                        help='Choose adversarial attack: fgsm, basic_iter, ll_class')    

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    transformation = transforms.ToTensor()
    # Dataset normalize
    if args.dataset_normalize:
        transformation = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('../data', train=True, download=True,
                        transform=transformation),
      batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('../data', train=False, download=True,
                        transform=transformation),
      batch_size=args.test_batch_size, shuffle=True)
  
    model = GoogLeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.test_mode is False:
        train(args, model, device, train_loader, optimizer)
        test(args, model, device, test_loader)
    else:
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()            

