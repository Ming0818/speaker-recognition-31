#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.cluster import KMeans 
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import cluster, mixture
from sklearn.manifold import TSNE as tsne
import matplotlib.pyplot as plt
import matplotlib
import pandas as pand


def sse(point, centroid):
	return sum(np.square(np.abs(np.subtract(point, centroid))))


def cluster_(all_dat, dps = ['tammany', 'jake', 'lindsay'], which_cluster=None, data=None):

	which_cluster = ['kmeans', 'ward_agglom', 'gaussian_means',
                     'spectral', 'birch'] if which_cluster is None else which_cluster
	all_sse = {}
	for k in range(2, 10):
		# if I were to use cosine distance from scipy, subtract that from 1 to get cosine similarity
		# kmeans already just uses euclidean distance
		kmeans = KMeans(n_clusters=k).fit(all_dat)
		clusters0 = kmeans.labels_
		centroids = kmeans.cluster_centers_
		curr_sse = 0

		for i in range(len(clusters0)):
			curr_sse += sse(X[i], centroids[clusters0[i]])

		all_sse[k] = curr_sse

		print "finished kmeans"
        #
		# #X = pd.DataFrame(all_dat, index=map(lambda x: x.split()[0],dps))
        #
		# # --------- setting up diff clustering options
		# ward = cluster.AgglomerativeClustering(
		# 	n_clusters=k, linkage='ward')
		# clusters1 = ward.fit_predict(all_dat)
		# print "finished ward"
		# gmm = mixture.GaussianMixture(
		# 	n_components=k, covariance_type='full')
		# gmm.fit(all_dat)
		# clusters2 = gmm.predict(all_dat)
		# print "finished gmm"
		# spectral = cluster.SpectralClustering(
		# 	n_clusters=k, eigen_solver='arpack',
		# 	affinity='cosine')
		# clusters3 = spectral.fit_predict(all_dat)
		# print "finished spectral"
		# birch = cluster.Birch(n_clusters=k)
		# clusters4 = birch.fit_predict(all_dat)
		# print "finished birch"
		# # ------- plot the result in 2D
		# pca = sklearnPCA(n_components=2) #2-dimensional PCA
		# t = tsne(n_components=2) # 2D TSNE
		# transformed = pand.DataFrame(pca.fit_transform(all_dat))
		# #transformed = transformed.drop('parasitemia_perc')
		# #del dps[9] # removing parasitemia % from datapoints as well for plotting purposes
		# cluster_names = {'kmeans': clusters0, 'ward_agglom': clusters1, 'gaussian_means': clusters2,
		# 				 'spectral': clusters3, 'birch': clusters4 }
        #


		# for key in filter(lambda x: x if x in which_cluster else None, cluster_names.keys()):
		# 	curr = cluster_names[key]
		# 	fig = plt.figure()#(figsize=(12,7))
		# 	ax = plt.subplot(111)
		# 	# getting colors for each cluster
		# 	colors = np.linspace(0, 1, k)
		# 	colors = plt.cm.gist_ncar(colors)
        #
        #
		# 	font = {'family' : 'normal',
		# 			'weight' : 'heavy',
		# 			'size'   : 14}
        #
		# 	matplotlib.rc('font', **font)
        #
		# 	# print transformed
		# 	print "\n", key
		# 	for i in range(0, len(dps)):
		# 		print curr[i], dps[i]
		# 	for i in range(0, len(dps)):
		# 		plt.scatter(transformed.iloc[i][0], transformed.iloc[i][1], label=dps[i] + "(" + str(curr[i]) + ")", c=colors[curr[i]], s=50)
		# 	box = ax.get_position()
		# 	ax.set_position([box.x0, box.y0 + box.height * 0.1,
		# 					 box.width, box.height * 0.9])
        #
		# 	# lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.3,1),
		# 	#               fancybox=True, shadow=True, ncol=2, borderaxespad=0.,prop={'size': 10}) # to get on bottom, use (.92, .4)
		# 	ax.legend(bbox_to_anchor=(-1, -1, 1, .102), loc=3, ncol=2, borderaxespad=0)
		# 	plt.xlabel("PCA 1")
		# 	plt.ylabel("PCA 2")
		# 	plt.title("Clustering: " + key + ", k is: " + str(k))
        #
		# 	plt.legend()
		# 	matplotlib.rc('font', **font)
		# 	plt.show()
        #
		# fig = plt.figure(figsize=(10,7))

	return all_sse

# want to identify how many speakers there are in given data 


def fbank():
	X = []


	# reading in tammany features
	(t_rate,t_sig) = wav.read("tammany-hobbit.wav")
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
	(jake_rate,jake_sig) = wav.read("jake-hobbit.wav")
	mfcc_feat_jake = mfcc(jake_sig,jake_rate)
	d_mfcc_feat_jake = delta(mfcc_feat_jake, 2)
	fbank_feat_jake = logfbank(jake_sig,jake_rate)

	fbank_jake = map(tuple, fbank_feat_jake)
	for row in fbank_jake:
		tammany = row
		X.append(row)
		labels[tammany] = 'jake'

	# reading in lindsay features
	(lind_rate,lind_sig) = wav.read("lindsay-hobbit.wav")
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
	(t_rate,t_sig) = wav.read("tammany-hobbit.wav")
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
	(jake_rate,jake_sig) = wav.read("jake-hobbit.wav")
	mfcc_feat_jake = mfcc(jake_sig,jake_rate)
	d_mfcc_feat_jake = delta(mfcc_feat_jake, 2)
	fbank_feat_jake = logfbank(jake_sig,jake_rate)

	mfcc_feat_jake = map(tuple, mfcc_feat_jake)
	for row in mfcc_feat_jake:
		tammany = row
		X.append(row)
		labels[tammany] = 'jake'

	# reading in lindsay features
	(lind_rate,lind_sig) = wav.read("lindsay-hobbit.wav")
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
	# print Counter(info)
	info = Counter(info)
	keys = sorted(info, key=lambda x: x[0])

	data = {'tammany': mfcc_feat_t, 'jake': mfcc_feat_jake, 'lind': mfcc_feat_lind}


	for k in keys:
		print k, info[k] / float(len(data[k[1]]))

	#return (keys, info)
	return X, data


X, data = mfcc_()
# print data
#
# print len(X)

sses = cluster_(X, which_cluster='kmeans')

vals = list(sses.values())
print sses.keys()
print vals

plt.plot([k for k in sses.keys()], [v for v in vals])
plt.show()