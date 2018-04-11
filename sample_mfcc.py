#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import Counter

# want to identify how many speakers there are in given data 


def fbank():
	X = []


	# reading in tammany features
	(t_rate,t_sig) = wav.read("tammany.wav")
	mfcc_feat_t = mfcc(t_sig,t_rate)
	d_mfcc_feat_t = delta(mfcc_feat_t, 2)
	fbank_feat_t = logfbank(t_sig,t_rate)

	labels = {}

	fbank_t = map(tuple, fbank_feat_t)

	for row in fbank_t:
		tammany = row
		X.append(row)
		labels[tammany] = 'tammany'

	# reading in jake features
	(jake_rate,jake_sig) = wav.read("jake.wav")
	mfcc_feat_jake = mfcc(jake_sig,jake_rate)
	d_mfcc_feat_jake = delta(mfcc_feat_jake, 2)
	fbank_feat_jake = logfbank(jake_sig,jake_rate)

	fbank_jake = map(tuple, fbank_feat_jake)
	for row in fbank_jake:
		tammany = row
		X.append(row)
		labels[tammany] = 'jake'

	# reading in lindsay features
	(lind_rate,lind_sig) = wav.read("lindsay.wav")
	mfcc_feat_lind = mfcc(lind_sig,lind_rate)
	d_mfcc_feat_lind = delta(mfcc_feat_lind, 2)
	fbank_feat_lind = logfbank(lind_sig,lind_rate)

	fbank_lind = map(tuple, fbank_feat_lind)

	for row in fbank_lind:
		tammany = row
		X.append(row)
		labels[tammany] = 'lind'


	kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
	print kmeans.labels_

	info = []

	for i in range(len(kmeans.labels_)):

		sample = X[i]
		name = labels[sample]
		info.append((kmeans.labels_[i], name))

	#print sorted(Counter(info), key=lambda x: x[0])
	info = Counter(info)
	keys = sorted(info, key=lambda x: x[0])

	data = {'tammany': len(fbank_t), 'jake': len(fbank_jake), 'lind': len(fbank_lind)}


	for k in keys:
		print k, info[k] / float(data[k[1]])

	return (keys, info)


def mfcc_():
	X = []

	# reading in tammany features
	(t_rate,t_sig) = wav.read("tammany.wav")
	mfcc_feat_t = mfcc(t_sig,t_rate)
	d_mfcc_feat_t = delta(mfcc_feat_t, 2)
	fbank_feat_t = logfbank(t_sig,t_rate)

	labels = {}

	mfcc_feat_t = map(tuple, mfcc_feat_t)

	for row in mfcc_feat_t:
		tammany = row
		X.append(row)
		labels[tammany] = 'tammany'

	# reading in jake features
	(jake_rate,jake_sig) = wav.read("jake.wav")
	mfcc_feat_jake = mfcc(jake_sig,jake_rate)
	d_mfcc_feat_jake = delta(mfcc_feat_jake, 2)
	fbank_feat_jake = logfbank(jake_sig,jake_rate)

	mfcc_feat_jake = map(tuple, mfcc_feat_jake)
	for row in mfcc_feat_jake:
		tammany = row
		X.append(row)
		labels[tammany] = 'jake'

	# reading in lindsay features
	(lind_rate,lind_sig) = wav.read("lindsay.wav")
	mfcc_feat_lind = mfcc(lind_sig,lind_rate)
	d_mfcc_feat_lind = delta(mfcc_feat_lind, 2)
	fbank_feat_lind = logfbank(lind_sig,lind_rate)

	mfcc_feat_lind = map(tuple, mfcc_feat_lind)

	for row in mfcc_feat_lind:
		tammany = row
		X.append(row)
		labels[tammany] = 'lind'


	kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
	print kmeans.labels_

	info = []

	for i in range(len(kmeans.labels_)):

		sample = X[i]
		name = labels[sample]
		info.append((kmeans.labels_[i], name))

	#print sorted(Counter(info), key=lambda x: x[0])
	print Counter(info)


fbank()

