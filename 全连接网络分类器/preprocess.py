import numpy as np


	# permutation = np.random.permutation(train_label.shape[0])
	# shuffled_dataset = train_data[permutation, :, :]
	# shuffled_labels = train_label[permutation]


def main():
	data_path = 'D:\\study\\研一下\\语音信号处理\\作业五\\3\\'
	speaker_names = np.array([])
	label = np.array([])
	for i in range(1, 17):
		with open(data_path+'ali\\ali'+str(i)+'.txt', 'r') as f:
			lines = f.readlines()
			for line in lines:
				name = ''
				for i in range(len(line)):
					if line[i] == ' ':
						break
				speaker_names = np.append(speaker_names, line[0:i])
				#speaker_names = np.append(speaker_names, line_split[0])
				label = np.append(label, line[i+1::])
	print(speaker_names)
	print(label.shape)

	raw_data = np.array([])
	ali_str = ''
	for i in range(1, 9):
		with open(data_path+'feats\\raw_mfcc'+str(i)+'.txt', 'r') as f:
			lines = f.readlines()
			for line in lines:
				#line = line.replace('\n', '')
				#print(line)
				if line[-1] == '\n':
					line = line[0:-1]
				if line[-2] == '[' or line[-1] == '[' or line == '':
					continue
				elif line[-1] == ']' or line[-2] == ']':
					ali_str += line[0:-1]
					raw_data = np.append(raw_data, ali_str)
					ali_str = ''
				else:
					ali_str += line+' '

	print(raw_data.shape)


	permutation = np.random.permutation(raw_data.shape[0])
	shuffled_dataset = raw_data[permutation]
	shuffled_labels = label[permutation]	

	cut_index = int(0.9 * raw_data.shape[0])
	print(cut_index)
	trainX = shuffled_dataset[0:cut_index]
	trainY = shuffled_labels[0:cut_index]
	testX = shuffled_dataset[cut_index::]
	testY = shuffled_labels[cut_index::]
	print(trainX.shape)
	print(trainY.shape)
	print(testX.shape)
	print(testY.shape)

	with open(data_path+'dataset\\trainX.txt', 'w') as f:
		for data_str in trainX:
			f.write(data_str)
	with open(data_path+'dataset\\testX.txt', 'w') as f:
		for data_str in testX:
			f.write(data_str)
	with open(data_path+'dataset\\trainY.txt', 'w') as f:
		for data_str in trainY:
			f.write(data_str)
	with open(data_path+'dataset\\testY.txt', 'w') as f:
		for data_str in testY:
			f.write(data_str)




if __name__=='__main__':
	main()
