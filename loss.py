import torch.nn as nn
from sobel import Sobel
import torch


def loss_rec(output, depth):
    ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)
    get_gradient = Sobel().to('cuda')
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
    loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    # print (loss_depth + loss_normal + (loss_dx + loss_dy))
    return loss_depth + loss_normal + (loss_dx + loss_dy)


def loss_depth(output, depth):
    # print(torch.log(torch.abs(output - depth) + 0.5).mean())
    return torch.log(torch.abs(output - depth) + 0.5).mean()


def loss_depth_grad(output, depth):
    get_gradient = Sobel().to('cuda')
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    return loss_depth + (loss_dx + loss_dy)


def loss_scale_invariant(output, depth):
    eps = 0.00000001
    # di = output - target
    di = torch.log(depth + eps) - torch.log(output + eps)
    n = (depth.size(2) * depth.size(3))
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
    loss = fisrt_term - second_term
    # print(loss.mean())
    return loss.mean()


def loss_mean_abs(output, depth):
    # print(torch.abs(depth - output).mean())
    return torch.abs(depth - output).mean()


def loss_mean_square(output, depth):
    d = depth - output
    return torch.pow(d, 2).mean()


def loss_custom(output, depth):
    eps = 0.00000001
    # di = output - target
    di = torch.log(depth + eps) - torch.log(output + eps)
    n = (depth.size(2) * depth.size(3))
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = 0.5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n ** 2)
    loss = fisrt_term - second_term
    # print(loss.mean())
    return loss.mean()


def loss_rec_plus_mean_abs(output, depth):
    ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)
    get_gradient = Sobel().to('cuda')
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
    loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    # print (loss_depth + loss_normal + (loss_dx + loss_dy))\
    loss_mean_abs = torch.abs(depth - output).mean()
    return loss_depth + loss_normal + (loss_dx + loss_dy) + loss_mean_abs


def loss_rec_with_mean_abs(output, depth):
    ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)
    get_gradient = Sobel().to('cuda')
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    # print (loss_depth + loss_normal + (loss_dx + loss_dy))\
    loss_mean_abs = torch.abs(depth - output).mean()
    return loss_mean_abs + loss_normal + (loss_dx + loss_dy)






def loss_abs_grad(output, depth):
    get_gradient = Sobel().to('cuda')
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    loss_abs = torch.abs(depth - output).mean()
    return loss_abs + (loss_dx + loss_dy)


def loss_rec_with_mean_square(output, depth):
    ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)
    get_gradient = Sobel().to('cuda')
    depth_grad = get_gradient(depth)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    d = depth - output
    loss_mean_square = torch.pow(d, 2).mean()
    return loss_mean_square + loss_normal + (loss_dx + loss_dy)
