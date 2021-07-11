import argparse
import auxil

from hyper_pytorch import *

import torch
from torch.autograd import Variable
import torch.nn.parallel
from torchvision.transforms import *

from model import CapsuleNet, caps_loss


def load_hyper(args):
    data, labels, numclass = auxil.loadData(args.dataset, num_components=args.components)
    pixels, labels = auxil.createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels = True)
    bands = pixels.shape[-1]; numberofclass = len(np.unique(labels))
    if args.tr_percent < 1: # split by percent
        x_train, x_test, y_train, y_test = auxil.split_data(pixels, labels, args.tr_percent)
    else: # split by samples per class
        x_train, x_test, y_train, y_test = auxil.split_data_fix(pixels, labels, args.tr_percent)
    if args.use_val: x_val, x_test, y_val, y_test = auxil.split_data(x_test, y_test, args.val_percent)
    del pixels, labels
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train))
    test_hyper  = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test))
    if args.use_val: val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"),y_val))
    else: val_hyper = None
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    val_loader  = torch.utils.data.DataLoader(val_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader, numberofclass, bands


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, num_classes, lam_recon):
    model.train()
    training_loss = 0; correct = 0
    for i, (x, y) in enumerate(trainloader):  # batch training
        y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
        x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

        optimizer.zero_grad()  # set gradients of optimizer to zero
        y_pred, x_recon = model(x, y)  # forward
        loss = caps_loss(y, y_pred, x, x_recon, lam_recon)  # compute loss
        loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
        training_loss += loss.item() * x.size(0)  # record the batch loss

        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()

        optimizer.step()  # update the trainable parameters with computed gradients

    return (training_loss / len(trainloader.dataset), correct.item() / len(trainloader.dataset))


def test(testloader, model, criterion, epoch, use_cuda, num_classes, lam_recon):
    model.eval()
    test_loss = 0; correct = 0
    for x, y in testloader:
        y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            y_pred, x_recon = model(x)
            test_loss += caps_loss(y, y_pred, x, x_recon, lam_recon).item() * x.size(0)  # sum up batch loss
            y_pred = y_pred.data.max(1)[1]
            y_true = y.data.max(1)[1]
            correct += y_pred.eq(y_true).cpu().sum()
    test_loss /= len(testloader.dataset)
    return test_loss, correct.item() / len(testloader.dataset)


def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda: inputs = inputs.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()] 
    return np.array(predicted)





def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='IP', type=str, help='dataset (options: IP, UP, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=100, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--te_bsize', default=100, type=int, help='mini-batch test size (default: 1000)')

    parser.add_argument('--lam_recon', default=0.0005, type=float, help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int, help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--spatialsize', dest='spatialsize', default=11, type=int, help='spatial-spectral patch dimension')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')

    parser.set_defaults(bottleneck=True)
    best_err1 = 100

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    train_loader, test_loader, val_loader, num_classes, n_bands = load_hyper(args)
    args.lam_recon = args.lam_recon*n_bands

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    if args.spatialsize < 9: avgpoosize = 1
    elif args.spatialsize <= 11: avgpoosize = 2
    elif args.spatialsize == 15: avgpoosize = 3
    elif args.spatialsize == 19: avgpoosize = 4
    elif args.spatialsize == 21: avgpoosize = 5
    elif args.spatialsize == 27: avgpoosize = 6
    elif args.spatialsize == 29: avgpoosize = 7
    else: print("spatialsize no tested")

    outpatches = {"5":1,"7":3,"9":5,"11":7, "13":9, "15":11, "17":13, "19":15, "21":17, "23":19, "27":23}
    outputsizepatch = outpatches[str(args.spatialsize)]
    model = CapsuleNet(input_size=[n_bands, args.spatialsize, args.spatialsize], classes=num_classes, routings=args.routings, outpatchdim=outputsizepatch)

    if use_cuda: model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1
    for epoch in range(args.epochs):
        #adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda, num_classes, args.lam_recon)
        if args.use_val: test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda, num_classes, args.lam_recon)
        else: test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda, num_classes, args.lam_recon)

        print("EPOCH", epoch, "TRAIN LOSS", train_loss, "TRAIN ACCURACY", train_acc, end=',')
        print("LOSS", test_loss, "ACCURACY", test_acc)
        # save model
        if test_acc > best_acc:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, "best_model.pth.tar")
            best_acc = test_acc

    checkpoint = torch.load("best_model.pth.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)
    print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)
    classification, confusion, results = auxil.reports(np.argmax(predict(test_loader, model, criterion, use_cuda), axis=1), np.array(test_loader.dataset.__labels__()), args.dataset)
    print(args.dataset, results)



if __name__ == '__main__':
	main()

