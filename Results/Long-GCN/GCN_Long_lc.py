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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from EMP.metrics import empCreditScoring, empChurn
from copy import deepcopy

print("[Checkpoint] All libraries imported successfully.")

# Configuration
class Config:
    DATA_DIR = r"/data/leuven/373/vsc37331/Mobile_Vikings/"
    # Call count edge files
    TRAIN_EDGE_C = "SN_M1t2_c.csv"
    VAL_EDGE_C = "SN_M2t3_c.csv"
    TEST_EDGE_C = "SN_M3t4_c.csv"
    # Call duration edge files
    TRAIN_EDGE_L = "SN_M1t2_l.csv"
    VAL_EDGE_L = "SN_M2t3_l.csv"
    TEST_EDGE_L = "SN_M3t4_l.csv"
    # Other data files
    TRAIN_LABEL = "L_M3.csv"
    TRAIN_RMF = "train_rmf_LT.csv"
    VAL_LABEL = "L_M4.csv"
    VAL_RMF = "val_rmf_LT.csv"
    TEST_LABEL = "L_test.csv"
    TEST_RMF = "test_rmf_LT.csv"
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
    def remove_nodes_and_create_data(rmf_path, edge_path_c, edge_path_l, label_path, churner_set_m1):
        """Create PyG data object with dual edge weights (call count and call duration)"""
        node_df = pd.read_csv(rmf_path)
        # Load both edge files - one for call count and one for duration
        edge_df_c = pd.read_csv(edge_path_c)
        edge_df_l = pd.read_csv(edge_path_l)
        label_df = pd.read_csv(label_path)
        print(f"[Checkpoint] Loaded RMF, both edge types (c and l), and label data")

        print(f"[Checkpoint] Starting remove_nodes_and_create_data")
        
        # Step 1: Map original node indices (1-based) to USR
        usr_list_old = node_df['USR'].tolist()
        index_to_usr = {idx: usr for idx, usr in enumerate(usr_list_old, start=1)}
        print(f"[Checkpoint] Mapped {len(index_to_usr)} node indices to USRs")

        # Step 2: No longer filtering out first month churners
        print(f"[Checkpoint] Keeping all users including first month churners")

        # Step 3: Map edge lists from index to USR for both edge types
        edge_df_c['i'] = edge_df_c['i'].map(index_to_usr)
        edge_df_c['j'] = edge_df_c['j'].map(index_to_usr)
        edge_df_l['i'] = edge_df_l['i'].map(index_to_usr)
        edge_df_l['j'] = edge_df_l['j'].map(index_to_usr)

        # Step 4: No longer filtering out edges with churners
        print(f"[Checkpoint] Keeping all edges including those involving first month churners")

        # Step 5: Map remaining USRs to new 0-based indices
        usr_list_new = node_df['USR'].tolist()
        usr_to_index = {usr: idx for idx, usr in enumerate(usr_list_new)}
        print(f"[Checkpoint] Created mapping for {len(usr_to_index)} users")

        # Step 6: Convert edge lists from USR to index and extract edge weights
        # For call count (c) edges
        mapped_i_c = edge_df_c['i'].map(usr_to_index)
        mapped_j_c = edge_df_c['j'].map(usr_to_index)
        edge_weights_c = edge_df_c['x'].values if 'x' in edge_df_c.columns else np.ones(len(mapped_i_c))
        
        # For call duration (l) edges
        mapped_i_l = edge_df_l['i'].map(usr_to_index)
        mapped_j_l = edge_df_l['j'].map(usr_to_index)
        edge_weights_l = edge_df_l['x'].values if 'x' in edge_df_l.columns else np.ones(len(mapped_i_l))
        
        print(f"[Checkpoint] Extracted call count weights with min={edge_weights_c.min()}, max={edge_weights_c.max()}")
        print(f"[Checkpoint] Extracted call duration weights with min={edge_weights_l.min()}, max={edge_weights_l.max()}")

        # Filter out any edges with NA mappings in both edge types
        valid_edges_c = ~(mapped_i_c.isna() | mapped_j_c.isna())
        mapped_i_c = mapped_i_c[valid_edges_c].values
        mapped_j_c = mapped_j_c[valid_edges_c].values
        edge_weights_c = edge_weights_c[valid_edges_c]
        
        valid_edges_l = ~(mapped_i_l.isna() | mapped_j_l.isna())
        mapped_i_l = mapped_i_l[valid_edges_l].values
        mapped_j_l = mapped_j_l[valid_edges_l].values
        edge_weights_l = edge_weights_l[valid_edges_l]

        # Merge edges from both edge types
        # First, create a dictionary with (i, j) as keys
        edge_dict = {}
        
        # Add call count edges
        for idx in range(len(mapped_i_c)):
            i, j = int(mapped_i_c[idx]), int(mapped_j_c[idx])
            count = float(edge_weights_c[idx])
            edge_dict[(i, j)] = [count, 0.0]  # [count, duration]
        
        # Add or update with call duration edges
        for idx in range(len(mapped_i_l)):
            i, j = int(mapped_i_l[idx]), int(mapped_j_l[idx])
            duration = float(edge_weights_l[idx])
            if (i, j) in edge_dict:
                edge_dict[(i, j)][1] = duration
            else:
                edge_dict[(i, j)] = [0.0, duration]
        
        # Convert merged edges to tensors
        edge_pairs = list(edge_dict.keys())
        edge_features = list(edge_dict.values())
        
        src_nodes = [pair[0] for pair in edge_pairs]
        dst_nodes = [pair[1] for pair in edge_pairs]
        
        # Create undirected graph by adding reverse edges
        edge_index_0 = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_index_1 = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
        edge_index = torch.cat([edge_index_0, edge_index_1], dim=1)
        
        # Create 2D edge attributes (count, duration)
        edge_attr_0 = torch.tensor(edge_features, dtype=torch.float)
        # Duplicate the edge features for the reverse edges
        edge_attr_1 = torch.tensor(edge_features, dtype=torch.float)
        edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=0)
        
        print(f"[Checkpoint] Created undirected edge index with shape: {edge_index.shape}")
        print(f"[Checkpoint] Created edge attributes with shape: {edge_attr.shape}")

        # Step 7: Prepare node features
        feature_df = node_df.drop(columns=['USR', 'churn'], errors='ignore')
        # Modified to remove features with '30' and '90' instead of '60' and '90'
        feature_df = feature_df.drop(columns=[col for col in feature_df.columns if '30' in col or '90' in col])
        print(f"[Checkpoint] Remaining feature columns: {list(feature_df.columns)}")

        scaler = StandardScaler()
        x = torch.tensor(scaler.fit_transform(feature_df), dtype=torch.float)
        print(f"[Checkpoint] Node feature tensor shape: {x.shape}")

        # Step 8: Prepare label tensor
        y = torch.tensor(label_df.iloc[:, -1].values, dtype=torch.long)
        print(f"[Checkpoint] Label tensor shape: {y.shape}")
        print(f"[Checkpoint] Label distribution: {torch.bincount(y)}")

        # Step 9: Create and return PyG Data object with edge weights
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr,  # 2D edge attributes [count, duration]
            y=y, 
            num_nodes=x.shape[0], 
            num_edges=edge_index.shape[1], 
            num_features=x.shape[1]
        )
        print(f"[Checkpoint] Created Data object with {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features, and 2D edge attributes")

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
        
        # Add edge feature transformation layer for 2D edge features
        self.edge_transform = nn.Linear(2, 1)  # Transform [count, duration] to a single weight
        nn.init.xavier_uniform_(self.edge_transform.weight, gain=0.1)
        
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

    def forward(self, x, edge_index, edge_attr=None):
        # Input validation
        if torch.isnan(x).any():
            print("[Checkpoint] WARNING: NaN detected in input features")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Process edge attributes if provided
        edge_weight = None
        if edge_attr is not None:
            # Combine call count and duration into a single edge weight
            # We can do this using our edge_transform layer or another method
            edge_weight = self.edge_transform(edge_attr).squeeze(-1)
            # Normalize edge weights to prevent numerical issues
            if edge_weight.max() > 1000:
                edge_weight = edge_weight / edge_weight.max()
            
        node_indices = torch.arange(x.size(0), device=x.device)
        node_emb = self.embedding(node_indices)
        feature_emb = self.feature_transform(x)
        
        combined = torch.cat([node_emb, feature_emb], dim=1)
        h = F.relu(self.combine(combined))  # Using ReLU for more stability than ELU
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        for conv in self.convs:
            # GCNConv in PyG has edge_weight as a named parameter, not a positional one
            h_new = conv(h, edge_index, edge_weight)
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
        
        # Handle edge attributes
        edge_attr = None
        if hasattr(data, 'edge_attr'):
            if data.edge_attr is not None:
                edge_attr = data.edge_attr.to(self.device)
        
        out = self.model(x, edge_index, edge_attr)
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
            
            # Handle edge attributes
            edge_attr = None
            if hasattr(data, 'edge_attr'):
                if data.edge_attr is not None:
                    edge_attr = data.edge_attr.to(self.device)
            
            try:
                out = self.model(x, edge_index, edge_attr)
                loss = self.criterion(out.squeeze(), y)

                # Handle potential NaN/Inf values
                probs = torch.sigmoid(out).squeeze().cpu().numpy()
                
                # Check for and handle NaN values
                if np.isnan(probs).any():
                    print("[Checkpoint] WARNING: NaN detected in probabilities, replacing with 0.5")
                    probs = np.nan_to_num(probs, nan=0.5)
                
                y_true = data.y.cpu().numpy()

                # Calculate AUC
                auc_roc = roc_auc_score(y_true, probs)
                
                # Calculate AUPRC
                precision, recall, _ = precision_recall_curve(y_true, probs)
                auprc = auc(recall, precision)
                
                # Calculate various lift metrics
                lift_0005 = self.calculate_lift(y_true, probs, 0.005)
                lift_001 = self.calculate_lift(y_true, probs, 0.01)
                lift_005 = self.calculate_lift(y_true, probs, 0.05)
                lift_01 = self.calculate_lift(y_true, probs, 0.1)

                metrics = {
                    'loss': loss.item(),
                    'auc': auc_roc,
                    'auprc': auprc,
                    'lift_0005': lift_0005,
                    'lift_001': lift_001,
                    'lift_005': lift_005,
                    'lift_01': lift_01,
                    'probs': probs,
                    'labels': y_true
                }
                
                if calculate_emp:
                    try:
                        emp_output = empChurn(probs, y_true, return_output=True, print_output=False)
                        metrics['emp'] = float(emp_output.EMP)
                        metrics['mp'] = float(emp_output.MP)
                    except Exception as e:
                        print(f"[Checkpoint] Error calculating EMP metrics: {str(e)}")
                        metrics['emp'] = 0.0
                        metrics['mp'] = 0.0
                else:
                    metrics['emp'] = 0.0
                    metrics['mp'] = 0.0
                
                return metrics
            except Exception as e:
                print(f"[Checkpoint] Error during evaluation: {str(e)}")
                # Return default metrics in case of error
                return {
                    'loss': float('inf'),
                    'auc': 0.5,
                    'auprc': 0.5,
                    'lift_0005': 1.0,
                    'lift_001': 1.0,
                    'lift_005': 1.0,
                    'lift_01': 1.0,
                    'emp': 0.0,
                    'mp': 0.0,
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

        print("[Checkpoint] Loading training data with RMF features and dual edge weights")
        self.data_train = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path_c=Config.TRAIN_EDGE_C,
            edge_path_l=Config.TRAIN_EDGE_L,
            label_path=Config.TRAIN_LABEL,
            rmf_path=Config.TRAIN_RMF,
            churner_set_m1=self.churned_users
        )
        GraphDataProcessor.get_class_distribution(self.data_train)
        
        print("[Checkpoint] Loading validation data with RMF features and dual edge weights")
        self.data_val = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path_c=Config.VAL_EDGE_C,
            edge_path_l=Config.VAL_EDGE_L,
            label_path=Config.VAL_LABEL,
            rmf_path=Config.VAL_RMF,
            churner_set_m1=self.churned_users
        )

        print("[Checkpoint] Loading test data with RMF features and dual edge weights")
        self.data_test = GraphDataProcessor.remove_nodes_and_create_data(
            edge_path_c=Config.TEST_EDGE_C,
            edge_path_l=Config.TEST_EDGE_L,
            label_path=Config.TEST_LABEL,
            rmf_path=Config.TEST_RMF,
            churner_set_m1=self.churned_users
        )
        
        print("[Checkpoint] Experiment initialization complete")

    def run_hyperparameter_tuning(self):
        print("[Checkpoint] ====== Starting Hyperparameter Tuning ======")
        
        # Only track best model by AUPRC now
        best_val_auprc = 0
        best_model = None
        best_config = None
        
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

                                current_best_val_auprc = 0
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
                                        print(f"[Checkpoint] Validation metrics - AUC: {val_metrics['auc']:.6f}, AUPRC: {val_metrics['auprc']:.6f}, Loss: {val_metrics['loss']:.6f}")
                                        print(f"[Checkpoint] Validation lifts - @0.5%: {val_metrics['lift_0005']:.6f}, @1%: {val_metrics['lift_001']:.6f}, @5%: {val_metrics['lift_005']:.6f}, @10%: {val_metrics['lift_01']:.6f}")

                                        # Update best model based on AUPRC improvement
                                        if val_metrics['auprc'] > best_val_auprc:
                                            print(f"[Checkpoint] New best model found! AUPRC: {val_metrics['auprc']:.6f} (previous best: {best_val_auprc:.6f})")
                                            best_val_auprc = val_metrics['auprc']
                                            best_model = deepcopy(model)
                                            best_config = (lr, hidden, num_layers, alpha, gamma)
                                        
                                        # Early stopping based on AUPRC
                                        if val_metrics['auprc'] > current_best_val_auprc:
                                            current_best_val_auprc = val_metrics['auprc']
                                            best_epoch = epoch
                                            epochs_no_improve = 0
                                        else:
                                            epochs_no_improve += 1
                                            if epochs_no_improve >= Config.PATIENCE:
                                                print(f"[Checkpoint] Early stopping at epoch {epoch} (no AUPRC improvement for {Config.PATIENCE} evaluations)")
                                                break
                                
                                print(f"[Checkpoint] Configuration completed. Best AUPRC: {current_best_val_auprc:.6f}")
                            except Exception as e:
                                print(f"[Checkpoint] Error during configuration {config_num-1}: {str(e)}")
                                print("[Checkpoint] Skipping to next configuration")
                                continue

        print("\n[Checkpoint] ====== Final Evaluation ======")
        
        # Evaluate the best model
        if best_model is not None:
            print(f"[Checkpoint] Evaluating best model")
            test_metrics = Trainer(best_model, Config.DEVICE).evaluate(self.data_test, calculate_emp=True)
            
            print(f"[Checkpoint] Best config (lr={best_config[0]}, hidden={best_config[1]}, layers={best_config[2]}, alpha={best_config[3]}, gamma={best_config[4]}):")
            print(f"[Checkpoint] Test AUPRC={test_metrics['auprc']:.6f}")
            print(f"[Checkpoint] Test AUC={test_metrics['auc']:.6f}")
            print(f"[Checkpoint] Test EMP={test_metrics['emp']:.6f}")
            print(f"[Checkpoint] Test MP={test_metrics['mp']:.6f}")
            print(f"[Checkpoint] Test Lift@0.5%={test_metrics['lift_0005']:.6f}")
            print(f"[Checkpoint] Test Lift@1%={test_metrics['lift_001']:.6f}")
            print(f"[Checkpoint] Test Lift@5%={test_metrics['lift_005']:.6f}")
            print(f"[Checkpoint] Test Lift@10%={test_metrics['lift_01']:.6f}")

            # Save model predictions for further analysis
            try:
                print("\n[Checkpoint] Saving best model predictions for analysis")
                
                # Create DataFrame with predictions
                predictions_df = pd.DataFrame({
                    'true_labels': test_metrics['labels'],
                    'predicted_probs': test_metrics['probs']
                })
                
                # Save to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                predictions_df.to_csv(f"best_model_predictions_{timestamp}.csv", index=False)
                print(f"[Checkpoint] Predictions saved as best_model_predictions_{timestamp}.csv")
            except Exception as e:
                print(f"[Checkpoint] Error saving predictions: {str(e)}")

        return best_model

if __name__ == "__main__":
    # Set manual seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("[Checkpoint] ====== Script Started ======")
    try:
        experiment = Experiment()
        best_model = experiment.run_hyperparameter_tuning()
        print("[Checkpoint] ====== Script Finished ======")
    except Exception as e:
        print(f"[Checkpoint] CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
