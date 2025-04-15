# [CHECKPOINT] Imports and Setup
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from EMP.metrics import empCreditScoring, empChurn
from copy import deepcopy

print("[Checkpoint] All libraries imported successfully.")

# Configuration
class Config:
    DATA_DIR = r"/data/leuven/373/vsc37331/Mobile_Vikings/"
    TRAIN_EDGE = "SN_M2_l.csv"
    TRAIN_LABEL = "L_M3.csv"
    TRAIN_RMF = "train_rmf.csv"
    VAL_EDGE = "SN_M3_l.csv"
    VAL_LABEL = "L_M4.csv"
    VAL_RMF = "val_rmf.csv"
    TEST_EDGE = "SN_M4_l.csv"
    TEST_LABEL = "L_test.csv"
    TEST_RMF = "test_rmf.csv"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LEARNING_RATES = [0.01, 0.001, 0.0001]
    HIDDEN_CHANNELS = [32, 128, 256]
    LAYERS = [1, 3]
    PATIENCE = 15
    MAX_EPOCHS = 200
    DROPOUT_RATE = 0.5
    EMBEDDING_DIM = 32
    # Focal Loss hyperparameters to tune
    FOCAL_ALPHAS = [0.5, 0.75]   # Prioritize positive class more
    FOCAL_GAMMAS = [1.0, 2.0]    # Focus on hard examples more
    CLIP_GRAD_NORM = 1.0


print(f"[Checkpoint] Configuration initialized")
print(f"[Checkpoint] Using device: {Config.DEVICE}")

os.chdir(Config.DATA_DIR)
print(f"[Checkpoint] Working directory set to: {os.getcwd()}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply torch.clamp to prevent numerical instability
        inputs = torch.clamp(inputs, min=-88, max=88)
        
        # Binary cross-entropy term
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute probabilities safely
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, min=1e-7, max=1-1e-7)  # Avoid 0 and 1 for numerical stability
        
        # Focal Loss modulating factor
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final loss
        focal_loss = alpha_weight * modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class GraphDataProcessor:
    @staticmethod
    def load_churned_users():
        """Load the list of users who churned in month 1"""
        try:
            churn_m1 = pd.read_csv("L_M1.csv")
            nodes_to_remove = churn_m1[churn_m1['churn_m1'] == 1]['USR'].values.astype('int64')
            churner_set_m1 = set(nodes_to_remove)
            print(f"[Checkpoint] Identified {len(churner_set_m1)} users who churned in month 1")
            return churner_set_m1
        except FileNotFoundError:
            print("[Checkpoint] WARNING: L_M1.csv not found. No users will be excluded.")
            return set()
        except Exception as e:
            print(f"[Checkpoint] ERROR loading churned users: {str(e)}")
            return set()

    @staticmethod
    def remove_nodes_and_create_data(rmf_path, edge_path, label_path, churner_set_m1):
        node_df = pd.read_csv(rmf_path)
        edge_df = pd.read_csv(edge_path)
        label_df = pd.read_csv(label_path)
        print(f"[Checkpoint] Loaded RMF, edge, and label data")

        print(f"[Checkpoint] Starting remove_nodes_and_create_data")
        
        # Step 1: Map original node indices (1-based) to USR
        usr_list_old = node_df['USR'].tolist()
        index_to_usr = {idx: usr for idx, usr in enumerate(usr_list_old, start=1)}
        print(f"[Checkpoint] Mapped {len(index_to_usr)} node indices to USRs")

        # Step 2: Filter out churners from node and label data
        before_nodes = len(node_df)
        node_df = node_df[~node_df['USR'].isin(churner_set_m1)].reset_index(drop=True)
        label_df = label_df[~label_df['USR'].isin(churner_set_m1)].reset_index(drop=True)
        after_nodes = len(node_df)
        print(f"[Checkpoint] Removed {before_nodes - after_nodes} churned users from node/label data")

        # Step 3: Map edge list from index to USR
        edge_df['i'] = edge_df['i'].map(index_to_usr)
        edge_df['j'] = edge_df['j'].map(index_to_usr)

        # Step 4: Filter out edges with churners
        before_edges = len(edge_df)
        edge_df = edge_df[~edge_df['i'].isin(churner_set_m1) & ~edge_df['j'].isin(churner_set_m1)]
        after_edges = len(edge_df)
        print(f"[Checkpoint] Removed {before_edges - after_edges} edges involving churners")

        # Step 5: Map remaining USRs to new 0-based indices
        usr_list_new = node_df['USR'].tolist()
        usr_to_index = {usr: idx for idx, usr in enumerate(usr_list_new)}
        print(f"[Checkpoint] Created mapping for {len(usr_to_index)} remaining users")

        # Step 6: Convert edge list from USR to index
        mapped_i = edge_df['i'].map(usr_to_index)
        mapped_j = edge_df['j'].map(usr_to_index)

        missing_i = mapped_i.isna().sum()
        missing_j = mapped_j.isna().sum()
        print(f"[Checkpoint] Missing i mappings: {missing_i}, Missing j mappings: {missing_j}")

        edge_index_0 = torch.tensor([mapped_i.values, mapped_j.values], dtype=torch.long)
        edge_index_1 = torch.tensor([mapped_j.values, mapped_i.values], dtype=torch.long)
        edge_index = torch.cat([edge_index_0, edge_index_1], dim=1)
        print(f"[Checkpoint] Created undirected edge index with shape: {edge_index.shape}")

        # Step 7: Prepare node features
        feature_df = node_df.drop(columns=['USR', 'churn'], errors='ignore')
        feature_df = feature_df.drop(columns=[col for col in feature_df.columns if '60' in col or '90' in col])
        print(f"[Checkpoint] Remaining feature columns: {list(feature_df.columns)}")

        scaler = StandardScaler()
        x = torch.tensor(scaler.fit_transform(feature_df), dtype=torch.float)
        print(f"[Checkpoint] Node feature tensor shape: {x.shape}")

        # Step 8: Prepare label tensor
        y = torch.tensor(label_df.iloc[:, -1].values, dtype=torch.long)
        print(f"[Checkpoint] Label tensor shape: {y.shape}")
        print(f"[Checkpoint] Label distribution: {torch.bincount(y)}")

        # Step 9: Create and return PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0], num_edges=edge_index.shape[1], num_features=x.shape[1])
        print(f"[Checkpoint] Created Data object with {data.num_nodes} nodes and {data.num_edges} edges, and {data.num_features} features")

        return data

    @staticmethod
    def get_class_distribution(data):
        """Returns class distribution for monitoring"""
        y = data.y
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        print(f"[Checkpoint] Class distribution - Positives: {num_pos}, Negatives: {num_neg}")
        return num_pos, num_neg

class EnhancedGCN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_channels, num_layers, num_nodes, dropout_rate=Config.DROPOUT_RATE):
        super().__init__()
        print(f"[Checkpoint] Initializing EnhancedGCN with {input_dim} input features, {embedding_dim} embedding dim, {hidden_channels} hidden channels, {num_layers} layers")
        
        # Initialize weights with smaller values to prevent exploding gradients
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.1)
        
        self.feature_transform = nn.Linear(input_dim, embedding_dim)
        nn.init.xavier_uniform_(self.feature_transform.weight, gain=0.1)
        
        self.combine = nn.Linear(embedding_dim * 2, hidden_channels)
        nn.init.xavier_uniform_(self.combine.weight, gain=0.1)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = GCNConv(hidden_channels, hidden_channels)
            # Initialize GCN layers with appropriate initialization
            nn.init.xavier_uniform_(conv.lin.weight, gain=0.1)
            self.convs.append(conv)
            
        self.lin = nn.Linear(hidden_channels, 1)
        nn.init.xavier_uniform_(self.lin.weight, gain=0.1)
        
        self.dropout_rate = dropout_rate
        
        # Add BatchNorm to help with training stability
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Checkpoint] Total parameters: {total_params}")

    def forward(self, x, edge_index):
        # Input validation
        if torch.isnan(x).any():
            print("[Checkpoint] WARNING: NaN detected in input features")
            x = torch.nan_to_num(x, nan=0.0)
            
        node_indices = torch.arange(x.size(0), device=x.device)
        node_emb = self.embedding(node_indices)
        feature_emb = self.feature_transform(x)
        
        combined = torch.cat([node_emb, feature_emb], dim=1)
        h = F.relu(self.combine(combined))  # Using ReLU for more stability than ELU
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        for conv in self.convs:
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)  # Using ReLU for stability
            h_new = self.batch_norm(h_new)  # Apply batch normalization
            h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)
            
            # Add residual connection for better gradient flow
            if h.shape == h_new.shape:
                h = h_new + h
            else:
                h = h_new
            
        # Ensure output doesn't have extreme values
        out = self.lin(h)
        return torch.clamp(out, min=-10, max=10)  # Prevent extreme values

class Trainer:
    def __init__(self, model, device, alpha=0.25, gamma=2.0):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)

    def train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.float().to(self.device)
        
        out = self.model(x, edge_index)
        loss = self.criterion(out.squeeze(), y)
        
        if not torch.isfinite(loss).all():
            print("[Checkpoint] WARNING: Non-finite loss detected, skipping backward pass")
            return float('nan')
            
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD_NORM)
        
        optimizer.step()
        
        return loss.item()

    def evaluate(self, data, calculate_emp=False):
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y.float().to(self.device)
            
            try:
                out = self.model(x, edge_index)
                loss = self.criterion(out.squeeze(), y)

                # Handle potential NaN/Inf values
                probs = torch.sigmoid(out).squeeze().cpu().numpy()
                
                # Check for and handle NaN values
                if np.isnan(probs).any():
                    print("[Checkpoint] WARNING: NaN detected in probabilities, replacing with 0.5")
                    probs = np.nan_to_num(probs, nan=0.5)
                
                y_true = data.y.cpu().numpy()

                auc = roc_auc_score(y_true, probs)
                lift_005 = self.calculate_lift(y_true, probs, 0.005)
                lift_05 = self.calculate_lift(y_true, probs, 0.05)

                metrics = {
                    'loss': loss.item(),
                    'auc': auc,
                    'lift_005': lift_005,
                    'lift_05': lift_05,
                    'probs': probs,
                    'labels': y_true
                }
                
                if calculate_emp:
                    try:
                        emp_output = empChurn(probs, y_true, return_output=True, print_output=False)
                        metrics['emp'] = float(emp_output.EMP)
                    except Exception as e:
                        print(f"[Checkpoint] Error calculating EMP: {str(e)}")
                        metrics['emp'] = 0.0
                else:
                    metrics['emp'] = 0.0
                
                return metrics
            except Exception as e:
                print(f"[Checkpoint] Error during evaluation: {str(e)}")
                # Return default metrics in case of error
                return {
                    'loss': float('inf'),
                    'auc': 0.5,
                    'lift_005': 1.0,
                    'lift_05': 1.0,
                    'emp': 0.0,
                    'probs': np.array([0.5]),
                    'labels': np.array([0])
                }

    @staticmethod
    def calculate_lift(y_true, y_prob, percentage):
        try:
            sorted_indices = np.argsort(y_prob)[::-1]
            n_top = max(1, int(len(y_true) * percentage))
            top_indices = sorted_indices[:n_top]
            top_positive = y_true[top_indices].sum()
            top_positive_rate = top_positive / n_top
            
            overall_positive = y_true.sum()
            overall_positive_rate = overall_positive / len(y_true)
            
            if overall_positive_rate == 0:
                return 1.0
            
            return top_positive_rate / overall_positive_rate
        except Exception as e:
            print(f"[Checkpoint] Error calculating lift: {str(e)}")
            return 1.0

class Experiment:
    def __init__(self):
        print("[Checkpoint] ====== Starting Experiment setup ======")
        
        self.churned_users = GraphDataProcessor.load_churned_users()

        print("[Checkpoint] Loading training data with RMF features")
        self.data_train = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path=Config.TRAIN_EDGE, 
            label_path=Config.TRAIN_LABEL,
            rmf_path=Config.TRAIN_RMF,
            churner_set_m1=self.churned_users
        )
        GraphDataProcessor.get_class_distribution(self.data_train)
        
        print("[Checkpoint] Loading validation data with RMF features")
        self.data_val = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path=Config.VAL_EDGE, 
            label_path=Config.VAL_LABEL,
            rmf_path=Config.VAL_RMF,
            churner_set_m1=self.churned_users
        )

        print("[Checkpoint] Loading test data with RMF features")
        self.data_test = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path=Config.TEST_EDGE, 
            label_path=Config.TEST_LABEL,
            rmf_path=Config.TEST_RMF,
            churner_set_m1=self.churned_users
        )
        
        print("[Checkpoint] Experiment initialization complete")

    def run_hyperparameter_tuning(self):
        print("[Checkpoint] ====== Starting Hyperparameter Tuning ======")
        
        best_metrics = {'val_auc': 0}
        best_models = {'auc': None}
        best_configs = {'auc': None}
        
        num_features = self.data_train.x.shape[1]
        print(f"[Checkpoint] Number of input features: {num_features}")

        print(f"[Checkpoint] Testing {len(Config.LEARNING_RATES)}×{len(Config.HIDDEN_CHANNELS)}×{len(Config.LAYERS)}×{len(Config.FOCAL_ALPHAS)}×{len(Config.FOCAL_GAMMAS)} combinations")
        
        config_num = 1
        total_configs = len(Config.LEARNING_RATES) * len(Config.HIDDEN_CHANNELS) * len(Config.LAYERS) * len(Config.FOCAL_ALPHAS) * len(Config.FOCAL_GAMMAS)
        
        for lr in Config.LEARNING_RATES:
            for hidden in Config.HIDDEN_CHANNELS:
                for num_layers in Config.LAYERS:
                    for alpha in Config.FOCAL_ALPHAS:
                        for gamma in Config.FOCAL_GAMMAS:
                            print(f"\n[Checkpoint] ====== Configuration {config_num}/{total_configs} ======")
                            print(f"[Checkpoint] Training with lr={lr}, hidden={hidden}, layers={num_layers}, alpha={alpha}, gamma={gamma}")
                            config_num += 1

                            try:
                                model = EnhancedGCN(
                                    input_dim=num_features,
                                    embedding_dim=Config.EMBEDDING_DIM,
                                    hidden_channels=hidden, 
                                    num_layers=num_layers,
                                    num_nodes=self.data_train.num_nodes
                                )
                                
                                trainer = Trainer(model, Config.DEVICE, alpha=alpha, gamma=gamma)
                                
                                # Use Adam with weight decay to prevent overfitting and improve stability
                                optimizer = torch.optim.Adam(
                                    model.parameters(), 
                                    lr=lr,
                                    weight_decay=1e-5  # Add weight decay
                                )

                                best_val_auc = 0
                                epochs_no_improve = 0
                                best_epoch = 0
                                nan_epochs = 0  # Count consecutive NaN epochs

                                print(f"[Checkpoint] Starting training for up to {Config.MAX_EPOCHS} epochs")
                                
                                for epoch in range(1, Config.MAX_EPOCHS + 1):
                                    print(f"\n[Checkpoint] --- Epoch {epoch}/{Config.MAX_EPOCHS} ---")
                                    
                                    loss = trainer.train(self.data_train, optimizer)
                                    
                                    if np.isnan(loss):
                                        nan_epochs += 1
                                        print(f"[Checkpoint] NaN loss detected, nan_epochs={nan_epochs}")
                                        if nan_epochs >= 3:  # Skip this configuration after 3 consecutive NaN epochs
                                            print("[Checkpoint] Too many NaN epochs, skipping configuration")
                                            break
                                    else:
                                        nan_epochs = 0  # Reset counter on successful epoch
                                        print(f"[Checkpoint] Epoch {epoch} training loss: {loss:.6f}")

                                    if epoch % 5 == 0 and not np.isnan(loss):
                                        val_metrics = trainer.evaluate(self.data_val)
                                        print(f"[Checkpoint] Validation metrics - AUC: {val_metrics['auc']:.6f}, Loss: {val_metrics['loss']:.6f}")
                                        print(f"[Checkpoint] Validation metrics - Lift@0.5%: {val_metrics['lift_005']:.6f}, Lift@5%: {val_metrics['lift_05']:.6f}")

                                        if val_metrics['auc'] > best_metrics['val_auc']:
                                            print(f"[Checkpoint] New best model found! AUC: {val_metrics['auc']:.6f}")
                                            best_metrics['val_auc'] = val_metrics['auc']
                                            best_models['auc'] = deepcopy(model)
                                            best_configs['auc'] = (lr, hidden, num_layers, alpha, gamma)
                                        
                                        if val_metrics['auc'] > best_val_auc:
                                            best_val_auc = val_metrics['auc']
                                            best_epoch = epoch
                                            epochs_no_improve = 0
                                        else:
                                            epochs_no_improve += 1
                                            if epochs_no_improve >= Config.PATIENCE:
                                                print(f"[Checkpoint] Early stopping at epoch {epoch}")
                                                break
                                
                                print(f"[Checkpoint] Configuration completed. Best AUC: {best_val_auc:.6f}")
                            except Exception as e:
                                print(f"[Checkpoint] Error during configuration {config_num-1}: {str(e)}")
                                print("[Checkpoint] Skipping to next configuration")
                                continue

        print("\n[Checkpoint] ====== Final Evaluation ======")
        for metric, model in best_models.items():
            if model is not None:
                print(f"[Checkpoint] Evaluating best model by {metric}")
                test_metrics = Trainer(model, Config.DEVICE).evaluate(self.data_test, calculate_emp=True)
                
                config = best_configs[metric]
                print(f"[Checkpoint] Best config (lr={config[0]}, hidden={config[1]}, layers={config[2]}, alpha={config[3]}, gamma={config[4]}):")
                print(f"[Checkpoint] Test AUC={test_metrics['auc']:.6f}")
                print(f"[Checkpoint] Test EMP={test_metrics['emp']:.6f}")
                print(f"[Checkpoint] Test Lift@0.5%={test_metrics['lift_005']:.6f}")
                print(f"[Checkpoint] Test Lift@5%={test_metrics['lift_05']:.6f}")

        return best_models

if __name__ == "__main__":
    # Set manual seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("[Checkpoint] ====== Script Started ======")
    try:
        experiment = Experiment()
        best_models = experiment.run_hyperparameter_tuning()
        print("[Checkpoint] ====== Script Finished ======")
    except Exception as e:
        print(f"[Checkpoint] CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()