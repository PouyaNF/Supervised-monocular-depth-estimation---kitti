import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import DataGenerator
import warnings
from fast_depth import MobileNetSkipAdd
warnings.filterwarnings('ignore')


def imshow(img):
    #img = img / 2 + 0.5     # un-normalize
    npimg = img.detach().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    if npimg.shape[2] == 1:
      plt.imshow(npimg.squeeze(axis=2) )
    else:
      plt.imshow(npimg,  vmin=0, vmax=255)
    plt.show()


def un_normalize(tensor, mean, std):
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.add_(m).mul_(s)
    return tensor


batch_size = 1
kitti_root_path = 'Dataset/Kitti'
datagen = DataGenerator(kitti_root_path, phase='train', high_gpu=True)
kittidataset = datagen.create_data(batch_size, nthreads=0)
depth_net = 'depthNet.pth'


def main():
    for i, data in enumerate(kittidataset):
        image, depth = data['img'], data['depth']
        print('batch' + str(i) + ':' + ' image tensor:' + str(data['img'].size()) +
              '  ', 'depth tensor:' + str(data['depth'].size()))

        model = MobileNetSkipAdd()
        model.to('cpu')
        checkpoint = torch.load(depth_net)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('all the keys matched')

        estimated_depth = model(image)
        if i == 0:
            break

    for i in range(batch_size):
        img = image[i]
        depth = depth[i]
        output = estimated_depth[i]
        #img = un_normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
        imshow(img)
        imshow(depth)
        imshow(output)


if __name__ == '__main__':
    main()








