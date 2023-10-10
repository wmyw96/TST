import numpy as np
import torch
import data as mydata
from model import RegressionNN
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import matplotlib.pyplot as plt
import argparse
import time
from colorama import init, Fore
from scipy.stats import t

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--T", help="number of samples", type=int, default=200)
parser.add_argument("--depth", help="number of layers", type=int, default=2)
parser.add_argument("--width", help="number of layers", type=int, default=64)
parser.add_argument("--window", help="window size", type=int, default=20)
parser.add_argument("--record_dir", help="directory to save record", type=str, default="")
parser.add_argument("--seed", help="number of layers", type=int, default=4869)
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--epochs", help="number of epochs", type=int, default=200)
parser.add_argument("--gvideo", help="generate video", type=bool, default=False)
parser.add_argument("--replicates", help="number of replications", type = int, default=200)

args = parser.parse_args()

ts_model = mydata.TimeSeriesSampler(5, mydata.sample_func1)

window_size = 5


def sequence_data(n_samples, n_omit):
	seq, gt = ts_model.sample(n_omit + n_samples + window_size)
	x, xgt = seq[n_omit:], gt[n_omit:]
	xs, ys, gts = [], [], []
	for i in range(n_samples):
		xs.append(x[i:i+window_size])
		ys.append(x[i+window_size])
		gts.append(xgt[i+window_size])
	xs = np.array(xs)
	ys = np.reshape(np.array(ys), (n_samples,1))
	return torch.tensor(xs).float(), torch.tensor(ys).float(), torch.tensor(gts).float()


def independent_data(n_samples, n_omit, seq_len):
	xs, ys = [], []
	for i in range(n_samples):
		v = np.random.randint(seq_len)
		x, _ = ts_model.sample(n_omit + v + 1 + window_size)
		xs.append(x[n_omit+v:n_omit+v+window_size])
		ys.append(x[n_omit+v+window_size])
	xs = np.array(xs)
	ys = np.reshape(np.array(ys), (n_samples,1))
	return torch.tensor(xs).float(), torch.tensor(ys).float()

## create dataset
n_train = args.T
n_valid = args.T * 3 // 7
n_test = args.T
batch_size = 32

x_train_dep, y_train_dep, _ = sequence_data(n_train, window_size)
x_valid_dep, y_valid_dep, _ = sequence_data(n_valid, window_size)
x_train_indep, y_train_indep = independent_data(n_train, window_size, n_train)
x_valid_indep, y_valid_indep = independent_data(n_valid, window_size, n_valid)
x_test, y_test, y_test_gt = sequence_data(n_test, window_size)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.kdeplot(ts_model.sample(n_train)[0], label='KDE of Sample Data')
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('density.pdf')
plt.close()


def create_torch_loader(x, y):
	dataset = torch.utils.data.TensorDataset(x, y)
	loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
	return loader


loader_dep = create_torch_loader(x_train_dep, y_train_dep)
loader_indep = create_torch_loader(x_train_indep, y_train_indep)
loaders = {'dep': loader_dep, 'indep': loader_indep}
valid = {'dep': (x_valid_dep, y_valid_dep), 'indep': (x_valid_indep, y_valid_indep)}

models, optimizers, schedulers = {}, {}, {}

settings = ['dep', 'indep']

for setting in settings:
	model = RegressionNN(window_size, args.depth, args.width)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs+1)
	models[setting], optimizers[setting], schedulers[setting] = model, optimizer, scheduler


errors = np.zeros((args.epochs, 6))
criterion = nn.MSELoss()

image_paths = []
if args.gvideo:
	if not os.path.exists("predictions_images"):
		os.makedirs("predictions_images")

errors_optimal = np.zeros((args.replicates,2))		
for i in range(args.replicates):
	for epoch in range(args.epochs):
		for setting in settings:
			loader, model, optimizer, scheduler = loaders[setting], models[setting], optimizers[setting], schedulers[setting]
			valid_data, valid_labels = valid[setting]

			model.train()
			train_losses = []

			for batch_idx, (data, target) in enumerate(loader):
				optimizer.zero_grad()
				output = model(data)
				loss = criterion(output, target)
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			
			model.eval()
			with torch.no_grad():
				valid_output = model(valid_data)
				test_output = model(x_test)
				valid_loss = criterion(valid_output, valid_labels)
				test_loss = criterion(test_output, y_test)

			offset = 0 + 3 * (setting == 'indep')
			errors[epoch, 0 + offset], errors[epoch, 1 + offset], errors[epoch, 2 + offset] = np.mean(train_losses), valid_loss.item(), test_loss.item()
			scheduler.step()
			if (epoch + 1) % 5 == 0:
				print(f'Epoch {epoch+1} [{setting}]: train = {np.mean(train_losses)}, valid = {valid_loss.item()}, test = {test_loss.item()}')
				ix = np.argmin(errors[:epoch+1, offset+1])
				print(f'[{setting}]: optimal valid error = {errors[ix, 1 + offset]}, corresponding test error = {errors[ix, 2 + offset]}')
				#save the optimal valid error

		if args.gvideo:
			# plot predictions
			models['dep'].eval()
			models['indep'].eval()
			with torch.no_grad():
				pred_dep = models['dep'](x_test)
				pred_indep = models['indep'](x_test)
			
			plt.figure(figsize=(14, 7))

			plt.title(f'epoch = {epoch}, valid loss = {test_loss.item()}')
				
			# Plot actual series
			plt.plot(np.arange(args.T), y_test_gt, label="Actual Training")
			# Plot predicted series
			plt.plot(np.arange(args.T), pred_dep.numpy()[:, 0], label="Dependent Data Predictions")
			plt.plot(np.arange(args.T), pred_indep.numpy()[:, 0], label="Independent Data Predictions")

			plt.xlabel("Time")
			plt.ylabel("Value")
			plt.legend()
			plt.grid(True)
				
			image_path = os.path.join("predictions_images", f"epoch_{epoch + 1}_predictions.png")
			plt.savefig(image_path)
			image_paths.append(image_path)
			plt.close()

		scheduler.step()

	idx_error_dep, idx_error_indep = np.argmin(errors[:,1]), np.argmin(errors[:,4])

	errors_optimal[i,0], errors_optimal[i,1] = errors[idx_error_dep,1], errors[idx_error_indep,4]

# t-test
alpha = 0.05
error_diff = errors_optimal[:,0] - errors_optimal[:,1]
std = np.std(error_diff)
Tstat = error_diff/(std/np.sqrt(args.epochs))
pval = 2*t.cdf(-np.abs(Tstat), df=args.epochs-1)
print(pval<0.05)

