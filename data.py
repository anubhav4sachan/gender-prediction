import numpy as np
import csv

def getDataset(path, labels):
	reader = csv.reader(open(path, 'r'))
	X = []
	y = []
	row_num = -1
	for row in reader:
		row_num += 1
		if row_num == 0:
			continue
        
		v = np.zeros(len(row))
		v[0] = float(row[0])						# writerID
		v[1] = float(row[1])						# pageID
		v[2] = 1.0 if row[2] == 'English' else 0.0	# Language (Is it in english?)
		v[3] = float(row[3])						# Same page?
		v[4:] = np.array(row[4:])
		
		X.append(v)
		y.append(labels[int(row[0])])

	return [np.array(X), np.array(y)]

def getLabels(path):
	reader = csv.reader(open(path, 'r'))
	labels = {}

	row_num = -1
	for row in reader:
		row_num += 1
		if row_num == 0:
			continue

		labels[int(row[0])] = int(row[1])

	return labels

def getLangData(X):
    arabic = []
    english = []
    
    for i in range(0, X.shape[0], 4):
        arabic.append(X[i,])
        arabic.append(X[i+1,])
        english.append(X[i+2,])
        english.append(X[i+3,])
        
    return [np.array(arabic), np.array(english)]

if __name__ == "__main__":
    
    pass