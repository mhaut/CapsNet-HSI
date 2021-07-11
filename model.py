"""
Pytorch implementation of CapsNet in paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       Launch `python CapsNet.py -h` for usage help

Result:
    Validation accuracy > 99.6% after 50 epochs.
    Speed: About 73s/epoch on a single GTX1070 GPU card and 43s/epoch on a GTX1080Ti GPU.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""
import random
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
from hyper_pytorch import *

class CapsuleNet(nn.Module):
	def __init__(self, input_size, classes, routings,outpatchdim):
		super(CapsuleNet, self).__init__()
		self.input_size = input_size
		self.classes = classes
		self.routings = routings
		ksize=3
		indimcap = 8
		inplanes = 256

		# Layer 1: Just a conventional Conv2D layer
		self.conv1 = nn.Conv2d(input_size[0], inplanes, kernel_size=ksize, stride=1, padding=0)
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm2d(inplanes)
		# Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
		self.primarycaps = PrimaryCapsule(inplanes, inplanes, indimcap, kernel_size=ksize, stride=1, padding=0)

		# Layer 3: Capsule layer. Routing algorithm works here.
		#self.digitcaps = DenseCapsule(in_num_caps=32*7*7, in_dim_caps=8,
									#out_num_caps=classes, out_dim_caps=16, routings=routings)
		self.digitcaps = DenseCapsule(in_num_caps=32*outpatchdim*outpatchdim, in_dim_caps=indimcap,
									out_num_caps=classes, out_dim_caps=indimcap*2, routings=routings)
		#pathsize=11
		#self.digitcaps = DenseCapsule(in_num_caps=32*4*4, in_dim_caps=8,
									#out_num_caps=classes, out_dim_caps=16, routings=routings)

		# Decoder network.
		#self.decoder = nn.Sequential(
			#nn.Linear(16*classes, 328),
			#nn.ReLU(inplace=True),
			#nn.Linear(328, 192),
			#nn.ReLU(inplace=True),
			#nn.Linear(192, input_size[0] * input_size[1] * input_size[2])#,
			#nn.Sigmoid()
		#)
		self.decoder = nn.Sequential(
			nn.Linear(16*classes, 328),
			nn.Sigmoid(),
			nn.Linear(328, 192),
			nn.Sigmoid(),
			nn.Linear(192, input_size[0] * input_size[1] * input_size[2])#,
			#nn.Sigmoid()
		)

	def forward(self, x, y=None):
		#x = self.bn(self.relu(self.conv1(x)))
		x = self.relu(self.bn(self.conv1(x)))
		x = self.primarycaps(x)
		x = self.digitcaps(x)
		length = x.norm(dim=-1)
		if y is None:  # during testing, no label given. create one-hot coding using `length`
			index = length.max(dim=1)[1]
			y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
		reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
		return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
	"""
	Capsule loss = Margin loss + lam_recon * reconstruction loss.
	:param y_true: true labels, one-hot coding, size=[batch, classes]
	:param y_pred: predicted labels by CapsNet, size=[batch, classes]
	:param x: input data, size=[batch, channels, width, height]
	:param x_recon: reconstructed data, size is same as `x`
	:param lam_recon: coefficient for reconstruction loss
	:return: Variable contains a scalar loss value.
	"""
	L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
		0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
	L_margin = L.sum(dim=1).mean()

	L_recon = nn.MSELoss()(x_recon, x)

	return L_margin + lam_recon * L_recon




def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        y = torch.zeros(y.size(0), args.num_classes).scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            x, y = Variable(x.cuda()), Variable(y.cuda())
            y_pred, x_recon = model(x)
            test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).item() * x.size(0)  # sum up batch loss
            y_pred = y_pred.data.max(1)[1]
            y_true = y.data.max(1)[1]
            correct += y_pred.eq(y_true).cpu().sum()
    test_loss /= len(test_loader.dataset)
    return test_loss, correct.item() / len(test_loader.dataset)

def teststats(model, test_loader, args):
    model.eval()
    if args.showim:
        predictions = []
        for x, y in test_loader:
            with torch.no_grad():
                x = Variable(x.cuda())
                y_pred, _ = model(x)
                y_pred = y_pred.data.max(1)[1]
                for yy in y_pred:
                    predictions.append(yy)
        predictions = np.array(predictions)
        return predictions
    else:
        predictions = []; real = []
        for x, y in test_loader:
            y = torch.zeros(y.size(0), args.num_classes).scatter_(1, y.view(-1, 1), 1.)
            with torch.no_grad():
                x, y = Variable(x.cuda()), Variable(y.cuda())
                y_pred, _ = model(x)
                y_pred = y_pred.data.max(1)[1]
                y_true = y.data.max(1)[1]
                for yy in y_pred:
                    predictions.append(yy)
                for yy in y_true:
                    real.append(yy)
        predictions = np.array(predictions)
        real = np.array(real)
        results = hl.metrics_classfier(real, predictions, range(len(np.unique(real))))[:-1]
        return results


def train(model, train_loader, test_loader, args):
	t0 = time()
	optimizer = Adam(model.parameters(), lr=args.lr)
	lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
	best_val_acc = 0.
	for epoch in range(args.epochs):
		model.train()  # set to training mode
		#if args.epochs >= 50 and args.epochs % 5 == 0:
			#lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
		ti = time()
		training_loss = 0.0
		for i, (x, y) in enumerate(train_loader):  # batch training
			#y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
			y = torch.zeros(y.size(0), args.num_classes).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
			x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

			optimizer.zero_grad()  # set gradients of optimizer to zero
			y_pred, x_recon = model(x, y)  # forward
			loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
			loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
			#training_loss += loss.data[0] * x.size(0)  # record the batch loss
			training_loss += loss.item() * x.size(0)  # record the batch loss
			optimizer.step()  # update the trainable parameters with computed gradients

		# compute validation loss and acc
		val_loss, val_acc = test(model, test_loader, args)
		logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
								val_loss=val_loss, val_acc=val_acc))
		print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
			% (epoch, training_loss / len(train_loader.dataset),
				val_loss, val_acc, time() - ti))
		if val_acc > best_val_acc:  # update best validation acc and save model
			best_val_acc = val_acc
			#torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
			print("best val_acc increased to %.4f" % best_val_acc)
			if args.idtest != None:
				torch.save(model.state_dict(), args.save_dir + '/trained_best_model_' + args.dataset + '_' + str(args.spatialsize) + '_' + str(args.idtest) +'.pkl')
			else:
				torch.save(model.state_dict(), args.save_dir + '/trained_best_model_' + args.dataset + '_' + str(args.spatialsize) + '.pkl')

	logfile.close()
	#torch.save(model.state_dict(), args.save_dir + '/trained_model_'+dataset+'_'+str(args.spatialsize)+'.pkl')
	#print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
	print("Total time = %ds" % (time() - t0))
	print('End Training' + '-' * 70)
	return model



def load_hyper(args):
	if args.testing:
		if args.dataset == 'indian' or args.dataset == 'pavia' or args.dataset == 'ksc' or args.dataset == 'salinas':
			type_norm = "standard"
			psize = args.spatialsize
			if args.dataset == 'indian':
				hsi_file = "/home/yann/software/hiperespectral/datasets/IndianPines/small_corrected/indian_pines_corrected.mat"
			elif args.dataset == 'pavia':
				hsi_file = "/media/yann/a79e5435-c7dc-4a01-a1ec-cd7ed8087ecb/datasets/Pavia/university/PaviaU.mat"
			elif args.dataset == 'ksc':
				hsi_file = "/media/yann/a79e5435-c7dc-4a01-a1ec-cd7ed8087ecb/datasets/KSC/KSC.mat"
			elif args.dataset == 'salinas':
				hsi_file = "/home/yann/software/hiperespectral/datasets/Salinas/salinas_corrected/salinas_corrected.mat"

			if args.showim:
				img_cubes, labels, _ = hl.read_hyperspectral_image_CNN(hsi_file, psize, type_norm, with_labels=True, filter_background=False)#, reduce_dim=)
				numberofclass = len(np.unique(labels)) - 1
			else:
				img_cubes, labels, _ = hl.read_hyperspectral_image_CNN(hsi_file, psize, type_norm, with_labels=True, filter_background=True)#, reduce_dim=)
				numberofclass = len(np.unique(labels))
			img_cubes = np.array(img_cubes)
			print("SHAPE INPUT", img_cubes.shape)
			bands = img_cubes.shape[-1]
			n_bands = img_cubes.shape[3] # input dimension

			img_cubes = np.transpose(img_cubes, (0, 3, 1, 2))
			final_hyper = HyperData((img_cubes, labels))

			kwargs = {'num_workers': 1, 'pin_memory': True}
			final_loader = torch.utils.data.DataLoader(final_hyper, batch_size=args.batch_size, shuffle=False, **kwargs)
			return None, final_loader, numberofclass, bands
		else:
			print("NO DATASET USED")
			exit()
	else:
		if args.dataset == 'indian' or args.dataset == 'pavia' or args.dataset == 'ksc' or args.dataset == 'salinas':
			type_norm = "standard"
			#type_norm = "minmax"
			psize = args.spatialsize
			if args.dataset == 'indian':
				hsi_file = "/home/yann/software/hiperespectral/datasets/IndianPines/small_corrected/indian_pines_corrected.mat"
				train_percent = [30,150,150,100,150,150,20,150,15,150,150,150,150,150,50,50]
				#train_percent = 0.15
				#train_percent = [50] * 16
			elif args.dataset == 'pavia':
				hsi_file = "/media/yann/a79e5435-c7dc-4a01-a1ec-cd7ed8087ecb/datasets/Pavia/university/PaviaU.mat"
				train_percent = [548,540,392,542,256,532,375,514,231]
				#train_percent = 0.15
				#train_percent = [50] * 9
			elif args.dataset == 'ksc':
				hsi_file = "/media/yann/a79e5435-c7dc-4a01-a1ec-cd7ed8087ecb/datasets/KSC/KSC.mat"
				train_percent = 0.20
				#train_percent = [153,49,52,51,33,46,21,87,104,81,84,101,186]
			elif args.dataset == 'salinas':
				hsi_file = "/home/yann/software/hiperespectral/datasets/Salinas/salinas_corrected/salinas_corrected.mat"
				train_percent = 0.15

			img_cubes, labels, _ = hl.read_hyperspectral_image_CNN(hsi_file, psize, type_norm, with_labels=True, filter_background=True)#, reduce_dim=1)
			img_cubes = np.array(img_cubes).astype("float32")
			numberofclass = len(np.unique(labels))
			print("SHAPE INPUT", img_cubes.shape)
			bands = img_cubes.shape[-1]
			n_bands = img_cubes.shape[3] # input dimension

			#rotate=True
			#if rotate:
				#img_cubes = np.array(img_cubes).astype("float32")
				#angles_rot = np.array([0, 90, 180, 270])
				#random_perm = np.array([random.randrange(angles_rot.size) for _ in range(img_cubes.shape[0])])
				#random_perm = np.array(angles_rot[random_perm])
				#for i,nrot in zip(range(img_cubes.shape[0]), random_perm):
					#img_cubes[i,:,:,:] = np.rot90(img_cubes[i,:,:,:], nrot, axes=(1,2))

			#x_train, x_test, y_train, y_test = hl.split_data_hiperlib(img_cubes, labels, training_percent=train_percent)
			x_train, x_test, y_train, y_test = hl.split_data_hiperlib(img_cubes, labels, training_select_pixels=train_percent)
			#x_test, _, y_test, _ = hl.split_data_hiperlib(x_test, y_test, training_percent=0.25, shuffle=False)
			del img_cubes, labels
			print(np.unique(y_train, return_counts=True))
			x_train   = np.transpose(x_train, (0, 3, 1, 2)).astype("float32")
			x_test    = np.transpose(x_test, (0, 3, 1, 2)).astype("float32")
			train_hyper = HyperData((x_train,y_train))
			test_hyper  = HyperData((x_test,y_test))

			kwargs = {'num_workers': 1, 'pin_memory': True}
			train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.batch_size, shuffle=True, **kwargs)
			test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.batch_size, shuffle=True, **kwargs)
			return train_loader, test_loader, numberofclass, bands
		else:
			print("NO DATASET USED")
			exit()

def load_mnist(path='./data', download=False, batch_size=100, shift_pixels=2):
	"""
	Construct dataloaders for training and test data. Data augmentation is also done here.
	:param path: file path of the dataset
	:param download: whether to download the original data
	:param batch_size: batch size
	:param shift_pixels: maximum number of pixels to shift in each direction
	:return: train_loader, test_loader
	"""
	kwargs = {'num_workers': 1, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(path, train=True, download=download,
					transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
													transforms.ToTensor()])),
		batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(path, train=False, download=download,
					transform=transforms.ToTensor()),
		batch_size=batch_size, shuffle=True, **kwargs)

	return train_loader, test_loader


if __name__ == "__main__":
	import argparse
	import os

	# setting the hyper parameters
	parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--batch_size', default=100, type=int)
	parser.add_argument('--lr', default=0.001, type=float,
						help="Initial learning rate")
	parser.add_argument('--lr_decay', default=0.9, type=float,
						help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
	parser.add_argument('--lam_recon', default=0.0005, type=float,
						help="The coefficient for the loss of decoder")
	parser.add_argument('-r', '--routings', default=3, type=int,
						help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
	parser.add_argument('-sps', '--spatialsize', default=3, type=int,
						help="Number of spatial size. should > 2")  # num_routing should > 0
	parser.add_argument('-idt', '--idtest', default=None, help="Id of test")  # num_routing should > 0
	parser.add_argument('-dat', '--dataset', default="indian", type=str,
						help="Name of dataset.")  # num_routing should > 0
	parser.add_argument('--shift_pixels', default=2, type=int,
						help="Number of pixels to shift at most in each direction.")
	parser.add_argument('--data_dir', default='./data',
						help="Directory of data. If no data, use \'--download\' flag to download it")
	parser.add_argument('--download', action='store_true',
						help="Download the required data.")
	parser.add_argument('--save_dir', default='./result')
	parser.add_argument('-t', '--testing', action='store_true',
						help="Test the trained model on testing dataset")
	parser.add_argument('-w', '--weights', default=None,
						help="The path of the saved weights. Should be specified when testing")
	parser.add_argument('-im', '--showim', default=None,
						help="mostrar imagen")
	args = parser.parse_args()
	args.showim = True if args.showim == "True" else False
	print(args)
	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	# load data
	loadmnist = False
	if loadmnist:
		args.spatialsize = 28
		args.num_classes = 10
		train_loader, test_loader = load_mnist(args.data_dir, download=True, batch_size=args.batch_size)
		args.lam_recon = args.lam_recon*784
		outputsizepatch = 7
	else:
		train_loader, test_loader, args.num_classes, nbands = load_hyper(args)
		args.lam_recon = args.lam_recon*nbands
		outpatches = {"5":1,"7":3,"9":5,"11":7, "13":9, "15":11, "17":13, "19":15, "21":17, "23":19, "27":23}
		outputsizepatch = outpatches[str(args.spatialsize)]
		#outputsizepatch = args.spatialsize -4

	# define model
	model = CapsuleNet(input_size=[nbands, args.spatialsize, args.spatialsize], classes=args.num_classes, routings=args.routings, outpatchdim=outputsizepatch)
	model.cuda()
	if args.testing == False:
		print(model)

	# train or test
	if args.weights is not None:  # init the model weights with provided one
		model.load_state_dict(torch.load(args.weights))
	if not args.testing:
		train(model, train_loader, test_loader, args)
	else:  # testing
		if args.weights is None:
			print('No weights are provided. Will test using random initialized weights.')

		if args.showim:
			data = teststats(model=model, test_loader=test_loader, args=args)
			np.savez("images/" + args.dataset + "_" + str(args.spatialsize) + "_" + str(args.idtest) + ".npz", np.array(data))
		else:
			#test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
			#print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
			#show_reconstruction(model, test_loader, 50, args)
			data = teststats(model=model, test_loader=test_loader, args=args)
			#nombres = ["OA","AA","Kappa","Classes"]
			#for aa in [3,0,1,2]:
				#print(nombres[aa], np.round(np.array(data[aa])*100, 2))
			print(args.dataset, args.spatialsize, np.round(data[0]*100,2), np.round(data[1]*100,2), np.round(data[2]*100,2), np.round(data[3]*100, 2).tolist(), sep="; ")

