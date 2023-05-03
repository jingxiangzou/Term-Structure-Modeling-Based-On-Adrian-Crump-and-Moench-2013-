import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


from datetime import datetime

import math

import torch as T
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from vit_pytorch import SimpleViT

import yfinance as yf

import matplotlib.pyplot as plt
from matplotlib import cm

from torch.utils.data import Dataset, DataLoader


def main():

	df = pd.read_excel("ACMTermPremium(3).xls", sheet_name='ACM Daily', index_col='DATE')
	df.index = pd.to_datetime(df.index)
	df = df.iloc[:,10:20]
	c = [i for i in range(10)]
	df.columns = c
	df = df[df.index >= pd.to_datetime("2000-01-01")]

	vix = yf.Ticker("^VIX").history(start="2000-01-01", end="2023-04-11",interval="1d")[["Close"]]
	vix.index = vix.index.tz_localize(None)
	data = pd.merge(df, vix, how='inner', left_index=True, right_index=True)
	

	X = np.array(data[9])
	Xp = np.array(data[9] + 2.)
	# X = np.exp(Xp) / 16 + 1 / Xp**3.3
	X = np.exp(X) / 16 + 1 / Xp
	Y = np.array(np.log(data["Close"]/100))
	X = X.reshape(-1,1)
	Y = Y.reshape(-1,1)

	reg = LinearRegression().fit(X, Y)
	print(reg.score(X,Y))
	print(reg.coef_, reg.intercept_)
	vix_hat = np.exp(reg.predict(X)) * 100
	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')
	plt.scatter(data[9], np.log(data["Close"]/100), label="VIX")
	plt.scatter(data[9], np.log(vix_hat/100), label="fitted")
	plt.legend()
	plt.show()

	plt.plot(data.index, data["Close"], label="VIX")
	plt.plot(data.index, vix_hat, label="fitted")
	plt.legend()
	plt.show()


	# batchsize = 64
	# Epochs = 1000
	# d = TPDataset(dataframe=data)
	# dataloader = DataLoader(d, batch_size=batchsize, shuffle=True)

	# device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
	# tpt = TPT().to(device)
	# # src_mask = T.from_numpy(np.zeros((batchsize,batchsize))).to(device).float()
	
	# L = nn.MSELoss()
	# optimizer = T.optim.Adam(tpt.parameters())

	# best = np.inf
	# insample = []
	# outsample = []

	# for i in range(Epochs):
	# 	l=0
	# 	for sample in dataloader:
	# 		# src_mask = T.from_numpy(np.zeros((len(sample[1]),len(sample[1])))).to(device).float()
	# 		r = tpt(sample[0].to(device))
	# 		loss = L(r, sample[1].to(device))
	# 		optimizer.zero_grad()
	# 		loss.backward()
	# 		optimizer.step()
	# 		l += loss.item()

	# 	insample += [l]
	# 	if l < best:
	# 		best = l
	# 		T.save(tpt.state_dict(), "./best_tpt.ckpt")

	# 	print(i,":",l, end='\r')

	# plt.plot(insample)
	# plt.show()

	# fitted = data[c].apply(lambda x: tpt(T.tensor([x]).float().to(device)).cpu().detach().numpy()[0], axis=1)

	# plt.plot(fitted)
	# plt.plot(data["Close"]/100)
	# plt.show()

	# print(np.corrcoef(curvature,data['Close']))
	# plt.scatter(curvature,data['Close'])
	# plt.xlabel('curvature')
	# plt.ylabel('VIX')
	# plt.show()

	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')

	# ax.scatter(data[9], curvature, data["Close"])
	# plt.show()

	exit(0)
	# X, Y = np.meshgrid(df.columns, np.arange(len(df.index)))
	# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	# surf = ax.plot_surface(X, Y, df, cmap=cm.turbo, antialiased=False)
	# plt.show()
	# exit(0)

	# prepare_data()
	# exit(0)

	# tpt = TPT()
	# print(tpt(T.tensor([[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.9]]),T.from_numpy(np.zeros((1,1)))))


	# exit(0)

	device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

	v = SimpleViT(
				image_size = 10,
				patch_size = 1,
				num_classes = 1,
				dim = 1024,
				depth = 6,
				heads = 16,
				mlp_dim = 2048,
				channels = 1
			)

	train_x, train_y, test_x, test_y = None, None, None, None

	with open('train-set.npy', 'rb') as f:
		train_x = np.load(f)
		train_y = np.load(f)

	print(train_y.shape)

	with open('test-set.npy', 'rb') as f:
		test_x = np.load(f)
		test_y = np.load(f)

	v.load_state_dict(T.load('./best(1).ckpt'))
	v.to(device)

	tx = np.array_split(train_x,50)
	fitted = []
	for i in tx:
		tmp = T.from_numpy(i).float().to(device)
		fitted += [v(tmp).cpu().detach().numpy()]

	fitted = np.squeeze(np.concatenate(fitted))
	vix = np.squeeze(train_y)
	# print(fitted)

	plt.plot(fitted, label="fitted")
	plt.plot(vix, label="VIX")
	plt.legend()
	plt.show()

	plt.plot(vix, fitted)
	plt.show()

	# v, inloss, outloss = train_vix(v, train_x, train_y, test_x, test_y)

	# T.save(v.state_dict(), './last.ckpt')

	# plt.plot(inloss, label="in sample")
	# plt.plot(outloss, label="out of sample")
	# plt.legend()
	# plt.show()


	return

def train_vix(v, train_x, train_y, test_x, test_y, batchsize=128, epochs=100, loss=None, device=None, verbose=True):

	if loss is None:
		loss = nn.MSELoss()
	if device is None:
		device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

	v = v.to(device)
	train_x = T.tensor(train_x).float()
	train_y = T.tensor(train_y).float()
	test_x = T.tensor(test_x).float()
	test_y = T.tensor(test_y).float()

	# batchsize = int(train_y.size()[0] // batchs_per_epoch)
	batchs_per_epoch = int(train_y.size()[0] // batchsize)
	tests_per_epoch = int(test_y.size()[0] // batchsize)

	# print(train_x.shape, train_y.shape, T.randn(1, 3, 256, 256).shape)

	opt = T.optim.Adam(v.parameters(), lr=3e-4)

	best = np.inf
	insample_loss = []
	outsample_loss = []
	for i in range(epochs):
		indices = T.randperm(train_y.size()[0])
		train_x = train_x[indices]
		train_y = train_y[indices]

		indices = T.randperm(test_y.size()[0])
		test_x = test_x[indices]
		test_y = test_y[indices]

		l = None
		for j in range(batchs_per_epoch):
			print("training -",j,end="\r")
			l = loss(v(train_x[j*batchsize:(j+1)*batchsize].to(device)), train_y[j*batchsize:(j+1)*batchsize].to(device))
			opt.zero_grad()
			l.backward()
			opt.step()

		l = 0
		for j in range(batchs_per_epoch):
			print("in sample test -",j,end="\r")
			l += loss(v(train_x[j*batchsize:(j+1)*batchsize].to(device)), train_y[j*batchsize:(j+1)*batchsize].to(device)).item()


		insample_loss += [l/batchs_per_epoch]

		l = 0
		for j in range(tests_per_epoch):
			print("out of sample test -",j,end="\r")
			l += loss(v(test_x[j*batchsize:(j+1)*batchsize].to(device)), test_y[j*batchsize:(j+1)*batchsize].to(device)).item()

		outsample_loss += [l/tests_per_epoch]


		if outsample_loss[-1] <= best:
			T.save(v.state_dict(), './best.ckpt')

		if verbose:
			print("Epoch:",i,"- insample-loss:", insample_loss[-1], ", outsample-loss:", outsample_loss[-1])


	return v, insample_loss, outsample_loss
	

def prepare_data():
	df = pd.read_excel("ACMTermPremium(3).xls", sheet_name='ACM Daily', index_col='DATE')
	df.index = pd.to_datetime(df.index)
	df = df.iloc[:,10:20]
	df.columns = [i for i in range(10)]
	
	vix = yf.Ticker("^VIX").history(period=None,interval = "1mo")
	vix.index = pd.to_datetime(vix.index)
	train_dates = vix.index[vix.index.values < np.datetime64('2020-01-01')]
	test_dates = vix.index[vix.index.values >= np.datetime64('2020-01-01')]
	
	train_x = []
	train_y = vix.loc[train_dates, 'Close']/100


	for i in train_dates:
		print(i, end="\r")
		train_x += [[df[df.index.values < i.replace(tzinfo=None)].tail(10)]]
		# train_y += [vix[i,'Close']]

	train_x = np.stack(train_x)
	train_y = np.reshape(np.array(train_y),(-1,1))


	test_x = []
	test_y = vix.loc[test_dates, 'Close']/100
	for i in test_dates:
		print(i, end="\r")
		test_x += [[df[df.index.values < i.replace(tzinfo=None)].tail(10)]]
		# train_y += [vix[i,'Close']]

	test_x = np.stack(test_x)
	test_y = np.reshape(np.array(test_y),(-1,1))

	with open('train-set.npy', 'wb') as f:
		np.save(f, train_x)
		np.save(f, train_y)

	with open('test-set.npy', 'wb') as f:
		np.save(f, test_x)
		np.save(f, test_y)

	return train_x, train_y, test_x, test_y


class TPT(nn.Module):

    def __init__(self, d_model: int=10, nhead: int=2, d_hid: int=64,
                 nlayers: int=3, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(1, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: T.Tensor) -> T.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # src = src * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # print(output.shape)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = T.arange(max_len).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = T.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = T.sin(position * div_term)
        pe[:, 0, 1::2] = T.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TPDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy().astype(np.float32)
        features = row[:-1]
        label = row[[-1]]/100
        return features, label

    def __len__(self):
        return len(self.dataframe.index)


if __name__ == '__main__':
	main()