# # Training set month 2: SN and month 3: labels

#create feature set from train_rmf_post.RData --> normalise!! + get rid of NaN
# Imports
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import itertools
from copy import deepcopy
from tqdm import tqdm
import logging
import os
from tabulate import tabulate
import matplotlib.pyplot as plt

# Set the working directory
os.chdir(r"/data/leuven/373/vsc37331/Mobile_Vikings/")
#get features from csv
node_attr= pd.read_csv("train_rmf.csv", sep=",",  header=0 )
node_attr.index += 1 #to make sure the nodes start counting at 1

nan_counts = node_attr.isna().sum()

print(nan_counts)


#don't need first column since this is USR 
node_attr= node_attr.iloc[:, 1:]
# List of columns you want to drop
columns_to_drop = ['M_60_on', 'M_90_on','F_60_on','F_90_on','numDialing_60_on','numDialed_60_on','numDialing_90_on','numDialed_90_on']
# Drop the specified columns
node_attr = node_attr.drop(columns=columns_to_drop)
# normalizing the attributes
scale = StandardScaler()
attrs_norm = scale.fit_transform(node_attr)

#to have more numbers after the comma
torch.set_printoptions(precision=10)

#now need to transform them to tensors
attrs_train = torch.tensor(attrs_norm, dtype=torch.float) #this can also be done in class

#create edge_index from SN data file

# Load your adjacency matrix as a NumPy array 
#first cvs to pd df then convert to numpy array
#depending on how cvs looks, need to define extra variables
adj_train = pd.read_csv("SN_M2_c.csv", sep=",",  header=0)

#make them into tensors
edge_idx_train_dir0=torch.tensor([adj_train['i'], adj_train['j']], dtype=torch.long)
edge_idx_train_dir1= torch.tensor([adj_train['j'], adj_train['i']], dtype=torch.long)
edge_idx_train= torch.cat((edge_idx_train_dir0, edge_idx_train_dir1), dim=1)

#create edge_attr from SN_ data file (don't normalise!!)   
#should we create tensor with only the attributes or also the edges?
edge_attrs_train_1= torch.tensor(adj_train['x'], dtype=torch.float64) 

#need it twice since we needed the edges in both directions
edge_attrs_train= torch.cat((edge_attrs_train_1,edge_attrs_train_1), dim=-1)
print(edge_attrs_train)


#create labels from L_ data file
#here we need to make sure that we read the right column (since there is also a column representing the USR)
labels_pd_train= pd.read_csv("L_M3.csv", sep=",",  header=0)

# now need to transform them to tensors
labels_train = torch.tensor(labels_pd_train['churn_m3'], dtype=torch.long)
print(labels_train) 

class MobileVikings_train:
    def __init__(self, attrs_train, edge_idx_train,edge_attrs_train,labels_train):
        
        self.data_train=Data(x=attrs_train, edge_index=edge_idx_train, edge_attr=edge_attrs_train,y=labels_train )
        
        self.data_train.num_classes=2
        self.data_train.num_features= len(node_attr.columns)
        

    
dataset= MobileVikings_train(attrs_train, edge_idx_train,edge_attrs_train,labels_train)

# we save this in data_train
data_train= dataset.data_train
print(data_train)


# # Validation month 3: SN and month 4: labels


#create edge_index from SN data file

# Load your adjacency matrix as a NumPy array 
#first cvs to pd df then convert to numpy array
#depending on how cvs looks, need to define extra variables
adj_val = pd.read_csv("SN_M3_c.csv", sep=",",  header=0)

#make them into tensors
edge_idx_val_dir0=torch.tensor([adj_val['i'], adj_val['j']], dtype=torch.long)
edge_idx_val_dir1= torch.tensor([adj_val['j'], adj_val['i']], dtype=torch.long)
# print(edge_idx_m4_dir0)
# print(edge_idx_m4_dir1)
edge_idx_val= torch.cat((edge_idx_val_dir0, edge_idx_val_dir1), dim=1)
print(edge_idx_val)

#create edge_attr from SN_ data file (don't normalise!!)  
#should we create tensor with only the attributes or also the edges?
edge_attrs_val_1= torch.tensor(adj_val['x'], dtype=torch.float64) 

#need it twice since we needed the edges in both directions
edge_attrs_val= torch.cat((edge_attrs_val_1,edge_attrs_val_1))
print(edge_attrs_val)


#create labels from L_ data file
#here we need to make sure that we read the right column (since there is also a column representing the USR)
labels_pd_val= pd.read_csv("L_M4.csv", sep=",",  header=0)

# now need to transform them to tensors
labels_val = torch.tensor(labels_pd_val['churn_m4'], dtype=torch.long)
print(labels_val) 



#it seems as if only creating a data object is enough since we don't need any of these extra methods (process, download,...)
#not sure since Data() is normally only for creating one graph

class MobileVikings_val:
    def __init__(self, attrs_train,edge_idx_val,edge_attrs_val,labels_val):
        #super(ProximusPost_m1, self).__init__('C:\\Users\\Hanna\\Documents\\GNN Churn\\Churn\\data_maria\\ProximusPost_csv', transform, pre_transform, pre_filter) 
        #transform, pre_transform and pre_filter are options --> can also put to None if don't need
        self.data_val=Data(x=attrs_train, edge_index=edge_idx_val, edge_attr=edge_attrs_val,y=labels_val )
        #self.data, self.slices = self.collate([data_m1]) #don't really get the use of this function
        self.data_val.num_classes=2
        self.data_val.num_features= len(node_attr.columns)
        

        #will need to define trainmask en testmask here as well


# #do we need those @properties too?
#     @property
#     def raw_file_names(self):
#         return 

#     @property
#     def processed_file_names(self):
#         return 
#     def process(self):
#         return
#     def download(self):
#         return 


dataset=MobileVikings_val(attrs_train,edge_idx_val,edge_attrs_val,labels_val)

#data=dataset.data_m1


#iniating the variable data_val, containing all our validation data
data_val = dataset.data_val
print(data_val)


# # Testing month 4: SN and month 5: labels


#create feature set from test_rmf_post.RData --> normalise!! + get rid of NaN (if there are any)

#get features from csv
test_node_attr= pd.read_csv("test_rmf.csv", sep=",",  header=0 )
test_node_attr.index += 1 #to make sure the nodes start counting at 1


# see whether there are nodes with NaN values for some features
nan_counts = test_node_attr.isna().sum()

print(nan_counts)
#there are no  NaN values

#don't need first column since they is USR 
test_node_attr= test_node_attr.iloc[:, 1:]

columns_to_drop = ['M_60_on', 'M_90_on','F_60_on','F_90_on','numDialing_60_on','numDialed_60_on','numDialing_90_on','numDialed_90_on']
# Drop the specified columns
test_node_attr = test_node_attr.drop(columns=columns_to_drop)

# normalizing the attributes
scale = StandardScaler()
test_attrs_norm = scale.fit_transform(test_node_attr)

#to have more numbers after the comma
torch.set_printoptions(precision=10)

#now need to transform them to tensors
attrs_test = torch.tensor(test_attrs_norm, dtype=torch.float) #this can also be done in class
print(attrs_test[-1]) 

#create edge_index from SN data file
# Load your adjacency matrix as a NumPy array 
#first cvs to pd df then convert to numpy array
#depending on how cvs looks, need to define extra variables
adj_test = pd.read_csv("SN_M4_c.csv", sep=",",  header=0)

edge_idx_test_dir0=torch.tensor([adj_test['i'], adj_test['j']], dtype=torch.long)
edge_idx_test_dir1= torch.tensor([adj_test['j'], adj_test['i']], dtype=torch.long)
# print(edge_idx_m5_dir0)
# print(edge_idx_m5_dir1)
edge_idx_test= torch.cat((edge_idx_test_dir0, edge_idx_test_dir1), dim=1)
print(edge_idx_test)

#create edge_attr from SN_ data file (don't normalise!!)  
#should we create tensor with only the attributes or also the edges?
edge_attrs_test_1= torch.tensor(adj_test['x'], dtype=torch.float64) 

#need it twice since we needed the edges in both directions
edge_attrs_test= torch.cat((edge_attrs_test_1,edge_attrs_test_1))
print(edge_attrs_test)

#create labels from L_ data file
#here we need to make sure that we read the right column (since there is also a column representing the )
labels_pd_test= pd.read_csv("L_test.csv", sep=",",  header=0)

#now need to transform them to tensors
labels_test = torch.tensor(labels_pd_test['churn_test'], dtype=torch.long)
print(labels_test)



#creating the dataset
from torch_geometric.data import InMemoryDataset, Data

class MobileVikings_test:
    def __init__(self, attrs_test, edge_idx_test,edge_attrs_test,labels_test):
        
        self.data_test=Data(x=attrs_test, edge_index=edge_idx_test, edge_attr=edge_attrs_test,y=labels_test )
        
        self.data_test.num_classes=2
        self.data_test.num_features= len(node_attr.columns)
        

    
dataset= MobileVikings_test(attrs_test, edge_idx_test,edge_attrs_test,labels_test)
data_test= dataset.data_test
print(data_test)


# # GCN_without looking at inbalance between churn-no churn



#defining our model

class GCN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, n_layers):
        super().__init__()
        self.conv1 = GCNConv(input_dim, embedding_dim)
        self.GCN_list = torch.nn.ModuleList([GCNConv(embedding_dim, embedding_dim)  for _ in range(n_layers-1)])
        self.lin = Linear(embedding_dim, 1)
        

    def forward(self,x, edge_index, edge_attr):
        #should i also add edge_attr above and below?
        h= self.conv1(x, edge_index, edge_weight = edge_attr)
        
        h= F.elu(h)
        
        h = F.dropout(h) #, training=self.training this checks whether the model is in training mode or not and sets the boolean accordingly 
        
        for l in self.GCN_list: #if 1 layer then this won't be executed, correct??
            h = l(h, edge_index, edge_weight = edge_attr)
            
            h = F.elu(h)
            h = F.dropout(h)
        
        h = self.lin(h)
        
        
        return h #don't do F.log_softmax(x, dim=1) since we will use CrossEntropyLoss as our loss function and this encapsulates also the softmax


#calculating the weights for the loss function
total_weight_class_0 = 0
total_weight_class_1 = 0

# Iterate over the first 4 months (since test data looks different)
for month in range(1, 5):
    labels_pd = pd.read_csv(f"L_M{month}.csv", sep=",", header=0)
    
    #weight_for_class_0 = labels_pd.shape[0] / ((labels_pd[f'churn_m{month}'] == 0).sum() * 2)
    weight_for_class_1 = labels_pd.shape[0] / ((labels_pd[f'churn_m{month}'] == 1).sum() * 2)
    
    # Add the weights to the total
    #total_weight_class_0 += weight_for_class_0
    total_weight_class_1 += weight_for_class_1

#only test month needs to be done seperately
labels_pd_m5 = pd.read_csv(f"L_test.csv", sep=",", header=0)
    
#weight_for_class_0_m5 = labels_pd.shape[0] / ((labels_pd_m5['churn_test'] == 0).sum() * 2)
weight_for_class_1_m5 = labels_pd.shape[0] / ((labels_pd_m5['churn_test'] == 1).sum() * 2)

#total_weight_class_0 +=  weight_for_class_0_m5
total_weight_class_1 += weight_for_class_1_m5

#avg_weight_for_class_0 = total_weight_class_0 / 5
avg_weight_for_class_1 = total_weight_class_1 / 5


# Create the weight tensor
weight = torch.tensor([  avg_weight_for_class_1])
print(weight)


# ## Defining the training function

#we used BCEWithLogitsLoss since then we only need to define the pos_weights (this will do BCELoss and sigmoid function so don't need sigmoid at end of forward pass)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
torch.manual_seed(332)

def train(model,lr):
    optimizer= torch.optim.Adam(model.parameters(), lr=lr) #can also add weight_decay
    model.train()
    optimizer.zero_grad() #if there were any gradients still saved, remove them
    out_train= model(data_train.x.float(), data_train.edge_index.type(torch.int64), data_train.edge_attr.float()) #here again need our name data + do we need to include edge_attr here?
    print(out_train.size())
    loss_train= criterion(out_train.flatten(), data_train.y.float()) 
    loss_train.backward()
    optimizer.step()
    return loss_train


# ## Defining the validation function
#creating a seperate validate funcion
@torch.no_grad()
def validate(model):
    model.eval()
    out_val= model(data_val.x.float(), data_val.edge_index.type(torch.int64), data_val.edge_attr.float()) 
    
    loss_val= criterion(out_val.flatten(), data_val.y.float()) 
    pred_probabilities = out_val.numpy()  
    true_labels = data_val.y.numpy()  
    # Calculate ROC AUC score for binary classification
    auc_score = roc_auc_score(true_labels, pred_probabilities)

    # Calculate Precision-Recall Curve and PR AUC
    precision, recall, _ = precision_recall_curve(true_labels, pred_probabilities)
    auprc = auc(recall, precision)
    return  loss_val, auc_score, auprc


# ## Defining the testing function


#testing the model based on data from month 5
#defining the testing function
@torch.no_grad() #this is so that the following block of code should be executed without tracking gradients
def test(model):
    model.eval()
    out_test= model(data_test.x.float(), data_test.edge_index.type(torch.int64), data_test.edge_attr.float())
    pred_probabilities = out_test.numpy()  #chatgpt suggested out_m5.cpu().numpy() but i don't think this cpu() is necessary
    true_labels = data_test.y.numpy()  #see above for cpu()
    # Calculate ROC AUC score for binary classification
    auc_score = roc_auc_score(true_labels, pred_probabilities)
    # Calculate Precision-Recall Curve and PR AUC
    precision, recall, _ = precision_recall_curve(true_labels, pred_probabilities)
    auprc = auc(recall, precision)
    return  auc_score, auprc

#code for early stopping when training the model, initiate the parameters patience and delta 
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

df_models= pd.DataFrame(columns= ['name model', 'val_auc', 'val_auprc', 'test_auc', 'test_auprc', 'best_epoch'])
print(df_models)


# ## Conclusion

# Code to loop through all possible hyperparameters

# #code to test only part of models
# #running through the training(with early stopping), validating and testing of the model 
# #each time with different values of the hyperparamaters 
# #and adding the validating and testing scores to the dataframe to then see which values for the parameters give the best results

#defining the different hyperparameters to check
learning_rates= [0.01,0.001,0.0001]
hidden_channels=[32,128,256]
layers= [1,3]
#set number of epochs high but use early stopping so that the model stops training after when the gain in validation_loss is very small
epochs= 500
best_e=0

# # Define the directory where you want to save the log file
log_directory = "Logging_GCN"

# # Ensure the log directory exists, create it if it doesn't
os.makedirs(log_directory, exist_ok=True)
# #create the logging file  
log_file = os.path.join(log_directory, "GCN_training_log.log")
# # Configure logging
logging.basicConfig(filename=log_file,
level=logging.INFO,
format='%(asctime)s - %(levelname)s: %(message)s')

# # Clear the log file at the beginning
with open(log_file, 'w'):
    pass

# # Create a dictionary to store training and validation losses for each model
model_losses = {}

# #making sure the dataframe is empty
df_models = df_models.iloc[0:0]

for lr_size, hidden_channels_size, number_of_layers in itertools.product(learning_rates,hidden_channels,layers):
    print(lr_size,hidden_channels_size,number_of_layers)
    current_model= GCN(data_train.num_features, hidden_channels_size, data_train.num_classes, number_of_layers)
    early_stopper = EarlyStopper(patience=10, min_delta=0.0005)
    logging.info(f"lr={lr_size}, hidden_channels={hidden_channels_size}, layers={number_of_layers}")
    best_e=0

#     # Create a key for the current model
    model_name= f"GCN_ST_MV_c_{number_of_layers}_layers_{hidden_channels_size}_embeddingDimension_{lr_size}_lr"
    model_losses[model_name] = {'train_loss': [], 'validation_loss': []}
    for epoch in tqdm(range (epochs+1)):
          train_loss = train(model=current_model,lr=lr_size)
          loss_val, val_auc, val_auprc = validate(model=current_model)
          validation_loss= loss_val.item() # Convert tensor to a Python scalar
          training_loss= train_loss.item() # Convert tensor to a Python scalar
          test_auc, test_auprc= test(model=current_model)
        
#         # Log the progress for each combination of hyperparameters
          logging.info(f"Epoch={epoch}")
          logging.info(f"Train Loss: {train_loss}, Validation Loss: {validation_loss}")
          logging.info(f"Validation AUC: {val_auc}, Validation AUPRC: {val_auprc}")
          logging.info(f"Test AUC: {test_auc}, Test AUPRC: {test_auprc}")

          # Append the training and validation losses to the dictionary
          model_losses[model_name]['train_loss'].append(training_loss)
          model_losses[model_name]['validation_loss'].append(validation_loss)

          #if epoch==10: #this is just to test in a quicker way without early stopping
          if epoch>20: #this is to overcome the overheating 
              #if epoch==21: #this is just to test in a quicker way without early stopping
              if early_stopper.early_stop(validation_loss):       #correct to say that if this condition is not satisfied we just continue with the next epoch        
                  best_model = deepcopy(current_model.state_dict()) 
                  best_val_auc= val_auc
                  best_val_auprc= val_auprc
                  best_test_auc= test_auc
                  best_test_auprc=test_auprc
                  best_e= epoch
                  model_name= f"GCN_ST_MV_c_{number_of_layers}_layers_{hidden_channels_size}_embeddingDimension_{lr_size}_lr"
                  new_data = {
                      'name model': model_name,
                      'val_auc': best_val_auc,
                      'val_auprc': best_val_auprc,
                      'test_auc': best_test_auc,
                      'test_auprc': best_test_auprc,
                      'best_epoch': best_e
                  }
                  new_row= pd.DataFrame(new_data, index=[0])
                 # Append the new row to the DataFrame
                  df_models = pd.concat([df_models,new_row],ignore_index=True )
                  #i think the issue lies within that it saves some kind of value and because of that the early stop condition is immedeatly true 
                  # and it doens't run for more epochs
                  break #is this a correct way of doing this???
          #continue

print(df_models)
table = tabulate(df_models, headers='keys', tablefmt='pretty', showindex=False)
print(table)

# Save DataFrame to CSV
df_models.to_csv("GCN.csv", index=False)
print("DataFrame saved as models_table.csv")

#creating code to visualize train_loss and val_loss of each model
# Create a folder to save the loss graphs
loss_graphs_directory = "LossGraphs_GCN"
os.makedirs(loss_graphs_directory, exist_ok=True)

# Remove existing files in the directory (if any)
existing_files = os.listdir(loss_graphs_directory)
for filename in existing_files:
    file_path = os.path.join(loss_graphs_directory, filename)
    os.remove(file_path)

# Iterate over each model and plot the losses
for model_name, losses in model_losses.items():
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses['train_loss'])), losses['train_loss'], label='Training Loss', linestyle='-')
    plt.plot(range(len(losses['validation_loss'])), losses['validation_loss'], label='Validation Loss', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Losses')
    plt.legend()
    
    # Save the graph as an image in the loss graphs directory
    graph_filename = os.path.join(loss_graphs_directory, f'{model_name}_losses.png')
    plt.savefig(graph_filename)
    plt.close()
    plt.show()


log_file_path = "Logging_GraphSAGE/GraphSAGE_training_log.log"

# Open the log file and read its contents
with open(log_file_path, "r") as log_file:
    content = log_file.read()
    print(content)

