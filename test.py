import torch
import torch.nn.parallel
import util
import numpy as np
import matplotlib.image
from dataset import DataGenerator
from fast_depth import MobileNetSkipAdd
import warnings
from numba import cuda
device = cuda.get_current_device()
device.reset()
warnings.filterwarnings('ignore')
matplotlib.rcParams['image.cmap'] = 'viridis'


def save_image(test_loader, dir):
    batch_size = 1

    for i, sample_batched in (enumerate(test_loader)):
        image, depth_ = sample_batched['img'], sample_batched['depth']
        print('batch' + str(i) + ':' + ' image tensor:' + str(sample_batched['img'].size()) +
              '  ', 'depth tensor:' + str(sample_batched['depth'].size()))
        image = image.to('cuda')
        depth_ = depth_.to('cuda')
        for j in range(batch_size):
            img = image[j]
            depth_ = depth_[j]
            img = img.squeeze().data.cpu().float().numpy()
            img = np.transpose(img, (1, 2, 0))
            depth_ = depth_.squeeze().data.cpu().float().numpy()

            matplotlib.image.imsave(dir + '/img' + str(i) + '.png', img)
            matplotlib.image.imsave(dir + '/depth' + str(i) + '.png', depth_)



def test(loader, model, dir):
    model.eval()
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    with torch.no_grad():
        for i, sample_batched in enumerate(loader):
            image, depth = sample_batched['img'], sample_batched['depth']
            print('batch' + str(i) + ':' + ' image tensor:' + str(sample_batched['img'].size()) +
                  '  ', 'depth tensor:' + str(sample_batched['depth'].size()))

            # image = torch.autograd.Variable(image, volatile=True).cuda()
            image = image.cuda()

            # non_blocking =  True :  if the source is in pinned memory, the copy will be asynchronous with respect to
            # the host. Otherwise, the argument has no effect. Default: False.
            depth = depth.cuda(non_blocking=True)

            output = model(image)
            # print(image.shape, depth.shape)

            batchSize = depth.size(0)
            errors = util.evaluateError(output, depth)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            totalNumber = totalNumber + batchSize
            averageError = util.averageErrors(errorSum, totalNumber)

            if i == 0 or i == 511 or i == 643 or i == 636 or i == 101 or \
                    i == 110 or i == 38 or i == 641:

                H = depth.size(2)
                W = depth.size(3)
                output = (output).squeeze().view(H, W).data.cpu().float().numpy()
                depth = (depth).squeeze().view(H, W).data.cpu().float().numpy()
                matplotlib.image.imsave(dir + '/prediction' + str(i) + '.png', output)
                matplotlib.image.imsave(dir + '/depth' + str(i) + '.png', depth)

    print('rmse:', np.sqrt(averageError['MSE']))
    print(averageError)





if __name__ == '__main__':

    depth_net = 'outputs/4-/checkpoint34.pth'
    save_to = 'outputs/4-/predictions'
    #save_to_test_images = 'outputs - other models/resnet/test images'
    kitti_root_path = 'D:/Pooya/Dataset/Kitti'

    ####### fast depth model
    model = MobileNetSkipAdd()
    checkpoint = torch.load(depth_net)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('all keys are matched : depth_net - fast_depth')
    model = model.cuda()

    # kitti loader
    batch_size = 1
    datagen = DataGenerator(kitti_root_path, phase='test', high_gpu=True)
    kittidataset = datagen.create_data(batch_size, nthreads=8)

    test(kittidataset, model, save_to)
    #save_image(kittidataset, save_to_test_images)




