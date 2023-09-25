import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def hello_simple_models():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from simple_models.py!')

def linear_classifier(data_dict,num_epochs=100,lr=0.1):
	X_train =  data_dict['X_train'].float()
	y_train = data_dict['y_train']
	X_val = data_dict['X_val'].float()
	y_val = data_dict['y_val']
	X_test = data_dict['X_test'].float()
	y_test = data_dict['y_test']
	# X_train = torch.tensor(X_train).clone().detach().float()
	# X_test = torch.tensor(X_test).clone().detach().float()
	# y_train = torch.tensor(y_train)
	# y_test = torch.tensor(y_test)
	# X_val = torch.tensor(X_train).clone().detach().float()
	# y_val = torch.tensor(y_train)
	  
	# Normalize the features
	mean = X_train.mean(dim=0)
	std = X_train.std(dim=0)
	X_train = (X_train - mean) / std
	X_test = (X_test - mean) / std
	X_val = (X_val - mean) / std


	model = torch.nn.Sequential(
	    torch.nn.Linear(in_features = 3072, out_features =10),
	    torch.nn.Softmax(dim=1)
	)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr)
	  
	num_epochs = num_epochs
	for epoch in range(num_epochs):
	    # Forward pass
		y_pred = model(X_train)
		loss = criterion(y_pred, y_train)
	  
	    # Backward pass and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	  
	    # Print the loss every 10 epochs
		if (epoch+1) % 10 == 0:
			with torch.no_grad():
				print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
				y_pred = model(X_train)
				_, predicted = torch.max(y_pred, dim=1)
				accuracy = (predicted == y_train).float().mean()
				print(f'Training Accuracy: {accuracy.item():.4f}')
				y_pred = model(X_val)
				_, predicted = torch.max(y_pred, dim=1)
				accuracy = (predicted == y_val).float().mean()
				print(f'Validation Accuracy: {accuracy.item():.4f}')

def knn(data_dict, neighbors):
	X_train =  data_dict['X_train'].cpu().detach().numpy()
	y_train = data_dict['y_train'].cpu().detach().numpy()
	X_val = data_dict['X_val'].cpu().detach().numpy()
	y_val = data_dict['y_val'].cpu().detach().numpy()
	X_test = data_dict['X_test'].cpu().detach().numpy()
	y_test = data_dict['y_test'].cpu().detach().numpy()

	knn=KNeighborsClassifier(n_neighbors=neighbors)
	knn.fit(X_train,y_train)
	y_pred_knn=knn.predict(X_val)
	accuracy = accuracy_score(y_pred_knn,y_val)
	print(f'Validation Accuracy: {accuracy.item():.4f}')
	



	    		
	
