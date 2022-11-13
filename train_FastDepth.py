import tqdm
import warnings
import matplotlib.pyplot as plt
from fast_depth import MobileNetSkipAdd
from dataset import DataGenerator
from loss import *
from numba import cuda
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping import EarlyStopping
device = cuda.get_current_device()
device.reset()
warnings.filterwarnings('ignore')



def plot_loss(train_loss, test_loss):
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss, color='blue', label='train loss')
    plt.plot(test_loss, color='red', label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('./output/train_loss.png')
    plt.show()




def train(model, dataloader, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    # Show progress in a Progress bar :
    for i, batch in tqdm.tqdm(enumerate(dataloader),
                              total=int(len(dataloader.dataset)) / dataloader.batch_size):
        # image, depth , new_depth= batch['img'],batch ['depth'], batch['depth_interp']
        image, depth = batch['img'], batch['depth']
        image = image.to('cuda')
        depth = depth.to('cuda')
        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, depth)
        train_running_loss += float(loss.item())
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(dataloader.dataset)
    return train_loss





def validate(model, dataloader, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader),
                                  total=int(len(dataloader.dataset)) / dataloader.batch_size):
            image, depth = batch['img'], batch['depth']
            image = image.to('cuda')
            depth = depth.to('cuda')
            output = model(image)
            loss = criterion(output, depth)
            val_running_loss += loss.item()
        val_loss = val_running_loss / len(dataloader.dataset)
        return val_loss




def main():
    depth_net = './output/pre -trained depth net- NYU/new_mobilenet-nnconv5dw-skipadd.pth '
    save_to = './output/depthNet.pth'

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('device :', device)

    kitti_root_path = 'D:/Pooya/Dataset/Kitti'
    batch_size = 16
    num_epoches = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    loss_type = loss_rec_with_mean_abs
    early_stopping_patience = 50
    ReduceLR_patience = 8

    # load model and optimizer
    model = MobileNetSkipAdd()
    checkpoint = torch.load(depth_net)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('all keys are matched : fast_depth')
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # print('optimizer params loaded')

    # data loaders
    # transformer will be defined automatically according to phase once datagen instance is created
    datagen_train = DataGenerator(kitti_root_path, phase='train', high_gpu=True)
    kittidataset_train = datagen_train.create_data(batch_size, nthreads=8)

    datagen_test = DataGenerator(kitti_root_path, phase='test', high_gpu=True)
    kittidataset_test = datagen_test.create_data(batch_size, nthreads=8)

    print('num of training examples:', len(kittidataset_train.dataset))
    print('num of validation examples:', len(kittidataset_test.dataset))

    # learning rate decay and early stopping initialising   -----------------
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=ReduceLR_patience, verbose=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=save_to)

    # training loop
    train_loss = []
    val_loss = []
    cudnn.benchmark = True
    for epoch in range(num_epoches):
        print(f"Epoch {epoch + 1} of {num_epoches}")
        ## train
        train_epoch_loss = train(model, kittidataset_train, optimizer, loss_type)
        train_loss.append(train_epoch_loss)
        print(f"Train Loss : {train_epoch_loss:.7f}")

        ## val
        val_epoch_loss = validate(model, kittidataset_test, loss_type)
        val_loss.append(val_epoch_loss)
        print(f"Val Loss : {val_epoch_loss:.7f}")

        scheduler.step(val_epoch_loss)
        early_stopping(val_epoch_loss, model, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if (epoch + 1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),  # This saves the trained neural network parameters.
                        'optimizer_state_dict': optimizer.state_dict(),  # save optimizer parameters
                        }, f'./output/fast_depth{epoch + 1}.pth')

    # plot_loss(train_loss)
    plot_loss(train_loss, val_loss)


if __name__ == '__main__':
    main()
