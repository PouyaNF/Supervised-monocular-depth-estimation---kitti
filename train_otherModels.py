import tqdm
import warnings
import matplotlib.pyplot as plt
from models import modules, net, resnet, densenet, senet
from dataset import DataGenerator
from loss import *
from numba import cuda
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
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


def train(model, dataloader, optimizer, criterion, scaler, gradient_accumulations):
    print('Training')
    model.train()
    train_running_loss = 0.0

    optimizer.zero_grad()
    # Show progress in a Progress bar :
    for i, batch in tqdm.tqdm(enumerate(dataloader),
                              total=int(len(dataloader.dataset)) / dataloader.batch_size):
        # image, depth, new_depth = batch['img'], batch['depth'] # if interp_method is ['linear', 'nyu']
        image, depth = batch['img'], batch['depth']  # if interp_method is ['nop'] in kittiloader.load_item
        image = image.to('cuda')
        depth = depth.to('cuda')
        # forward calculation of loss
        with autocast():  # automatic percision allocation to increase the speed
            output = model(image)
            loss = criterion(output, depth)
        train_running_loss += float(loss.item())
        scaler.scale(loss / gradient_accumulations).backward()

        # backward calculation of gradients
        if (i + 1) % gradient_accumulations == 0:  # accumulation of gradients gradient_accumulations times
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # train loss will be calculated for real batch size
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
            with autocast():
                output = model(image)
                loss = criterion(output, depth)
            val_running_loss += loss.item()
        val_loss = val_running_loss / len(dataloader.dataset)
        return val_loss


def experiment(encoder, learning_rate, num_epoches, loss_type, model_save_path, retrain, pre_trained_model='None'):
    kitti_root_path = 'Dataset/Kitti'
    batch_size = 16
    high_gpu = True
    gradient_accumulations = 1
    early_stopping_patience = 15
    ReduceLR_patience = 6
    nthreads = 8
    weight_decay = 1e-4


    # depth model ########################################################
    if encoder == 'resnet':
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    if encoder == 'densenet':
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])

    if encoder == 'senet':
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.cuda()

    if retrain:
        checkpoint = torch.load(pre_trained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('all keys are matched : depth net')

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('optimizer params are loaded')

    ############### data loader
    # transformer will be defined automatically according to phase once datagen instance is created
    datagen_train = DataGenerator(kitti_root_path, phase='train', high_gpu=high_gpu)
    kittidataset_train = datagen_train.create_data(batch_size, nthreads=nthreads)

    datagen_test = DataGenerator(kitti_root_path, phase='test', high_gpu=high_gpu)
    kittidataset_test = datagen_test.create_data(batch_size, nthreads=nthreads)

    print('num of training examples:', len(kittidataset_train.dataset))
    print('num of validation examples:', len(kittidataset_test.dataset))

    ####################### learning rate decay and early stopping initialising   -----------------
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=ReduceLR_patience, verbose=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=model_save_path)

    # Training loop #########################
    # auto cast
    scaler = GradScaler()
    cudnn.benchmark = True
    train_loss = []
    val_loss = []
    for epoch in range(num_epoches):
        print(f"Epoch {epoch + 1} of {num_epoches}")
        ## train
        train_epoch_loss = train(model, kittidataset_train, optimizer, loss_type, scaler, gradient_accumulations)
        train_loss.append(train_epoch_loss)
        print(f"Train Loss : {train_epoch_loss:.5f}")

        ## val
        val_epoch_loss = validate(model, kittidataset_test, loss_type)
        val_loss.append(val_epoch_loss)
        print(f"Val Loss : {val_epoch_loss:.5f}")

        scheduler.step(val_epoch_loss)
        early_stopping(val_epoch_loss, model, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # torch.save({'model_state_dict': model.state_dict(),  # This saves the trained neural network parameters.
        #            'optimizer_state_dict': optimizer.state_dict(),  # save optimizer parameters
        #            }, f'./output/fast_depth{epoch + 1}.pth')

    plot_loss(train_loss, val_loss)


if __name__ == '__main__':
    # load_depth_model = './output/1-loss depth- batch32-100 epochs/checkpoint.pth'
    loss_type = loss_rec_with_mean_abs
    num_epoches = 50
    learning_rate = 0.001
    model_save_path = 'outputs/senet/senet.pth'
    encoder = 'senet'
    experiment(encoder, learning_rate, num_epoches, loss_type, model_save_path, retrain=False, pre_trained_model='None')

    # loading pretrained model for senet got cuda low memory error
    # training densenet is not possible because of cuda low memory