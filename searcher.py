# importing the needed bindings
import numpy as np
import csv
 
class Searcher:

	def __init__(self, hsv, texture, tree, lp, lab, YCrCb):
		# saving the given index paths for hsv, tree and texture feature files
		self.hsv = hsv
		self.tree = tree
		self.texture = texture
		self.lp = lp
		self.lab = lab
		self.YCrCb = YCrCb

	def search(self, queryFeats, queryTexture, queryTree, queryLp, querylab, queryYCrCb, limit = 10):
		# initializing dictionaries to save results of color, texture
		# and tree matching
		finresult, cresult, txresult, tresult, lpresult, labresult, YCrCbresult= {}, {}, {}, {}, {}, {}, {}

		with open(self.hsv) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				cfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				cdis = self.chi_sqrd_distance(cfeats, queryFeats)
 
				# with image ID as key and distance as value, udpating the result dictionary
				cresult[record[0]] = cdis
				
			# closing the reader
			p.close()

		with open(self.texture) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				txfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				txdis = self.chi_sqrd_distance(txfeats, queryTexture)
 				
				# with image ID as key and distance as value, udpating the result dictionary
				txresult[record[0]] = txdis
				
			# closing the reader
			p.close()

		with open(self.tree) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				tfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				tdis = self.chi_sqrd_distance(tfeats, queryTree)
 				
				# with image ID as key and distance as value, udpating the result dictionary
				tresult[record[0]] = tdis
 
			# closing the reader
			p.close()

		with open(self.lp) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				lfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				ldis = self.chi_sqrd_distance(lfeats, queryLp)
 				
				# with image ID as key and distance as value, udpating the result dictionary
				lpresult[record[0]] = ldis
 
			# closing the reader
			p.close()
			
		with open(self.lab) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				labfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				labdis = self.chi_sqrd_distance(labfeats, querylab)
 				
				# with image ID as key and distance as value, udpating the result dictionary
				labresult[record[0]] = labdis
				
			# closing the reader
			p.close()	
			
		with open(self.YCrCb) as p:
			# opening the index path for reading
			read = csv.reader(p)
 
			# iterating over the records in the index
			for record in read:
				# spliting out the image ID and features
				YCrCbfeats = [float(x) for x in record[1:]]
				# calculating chi-squared distance between the index features and query features
				YCrCbdis = self.chi_sqrd_distance(YCrCbfeats, queryYCrCb)
 				
				# with image ID as key and distance as value, udpating the result dictionary
				YCrCbresult[record[0]] = YCrCbdis
				
			# closing the reader
			p.close()

		# iterating over all images in all results, combining their ranks from 
		# color, texture and tree comparisons
		for (k, v) in cresult.items():
			# getting a weighted sum of ranks for the image
			finresult[k] = (cresult[k]*0.25) + (tresult[k]*0.25) + (txresult[k]*0.0) + (lpresult[k]*0.50) + (labresult[k]*0.0) + + (YCrCbresult[k]*0.0)
 
		# sorting our results, in the increasing order of distances
		finresult = sorted([(v, k) for (k, v) in finresult.items()])
 
		# returning our results within the specified limit
		return finresult[:limit]

	def chi_sqrd_distance(self, histA, histB, eps = 1e-10):
		# calculating the chi-squared distance
		dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
 
		# returning the calculated chi-squared distance
		return dist
