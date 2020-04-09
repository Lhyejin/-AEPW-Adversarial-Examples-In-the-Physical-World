import torch
import torch.nn as nn
from torch.autograd import Variable

'''
White-Box Adversarial Attack
'''


# Fast Gradient sgin method
def fgsm(args, model, device, images, targets, criterion):

    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()

    # Generate perturbation
    grad_j = torch.sign(images.grad.data)
    adv_images = images + args.epsilon * grad_j
    lower_adv_images = torch.max(torch.tensor(0.).to(device),torch.max(images-args.epsilon, adv_images))
    adv_images = torch.min(torch.tensor(1.).to(device), torch.min(images+args.epsilon, lower_adv_images))
    adv_images = Variable(adv_images)
 
    return adv_images

# Basic Iteration method
def basic_iteration(args, model, device, images, targets, criterion):

    # because of image value [0, 1], args.epsilon * 255
    # number of iteration
    num_iter = int(min(args.epsilon*255 +4, 1.25*args.epsilon*255))

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

    return num_iter, adv_images

# Iterative Least-Likely Class Method
def least_likely_class(args, model, device, images, ll_targets, criterion):

    # image value [0, 1] args.epsilon * 255
    # number of iteration
    num_iter = int(min(args.epsilon*255+4, 1.25*args.epsilon*255))

    # X_0 adv images
    adv_images = images
    for i in range(num_iter):
        outputs = model(adv_images)
        loss = criterion(outputs, ll_targets)
        loss.backward()

        # Generate perturbation
        grad_j = torch.sign(adv_images.grad.data)
        next_adv_images = adv_images - args.epsilon * grad_j
        lower_adv_images = torch.max(torch.tensor(0.).to(device), torch.max(adv_images-args.epsilon, next_adv_images))
        adv_images = torch.min(torch.tensor(1.).to(device), torch.min(adv_images+args.epsilon, lower_adv_images))
        adv_images = Variable(adv_images.to(device), requires_grad=True)


    return num_iter, adv_images
