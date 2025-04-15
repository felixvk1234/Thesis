
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
    # Call graph files
    TRAIN_EDGE_CALL = "SN_M2_c.csv"
    VAL_EDGE_CALL = "SN_M3_c.csv"
    TEST_EDGE_CALL = "SN_M4_c.csv"
    # Location graph files
    TRAIN_EDGE_LOC = "SN_M2_l.csv"
    VAL_EDGE_LOC = "SN_M3_l.csv"
    TEST_EDGE_LOC = "SN_M4_l.csv"
    # Label files
    TRAIN_LABEL = "L_M3.csv"
    VAL_LABEL = "L_M4.csv"
    TEST_LABEL = "L_test.csv"
    # RMF features
    TRAIN_RMF = "train_rmf.csv"
    VAL_RMF = "val_rmf.csv"
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
    def remove_nodes_and_create_data(rmf_path, edge_call_path, edge_loc_path, label_path, churner_set_m1):
        node_df = pd.read_csv(rmf_path)
        edge_call_df = pd.read_csv(edge_call_path)
        edge_loc_df = pd.read_csv(edge_loc_path)
        label_df = pd.read_csv(label_path)
        print(f"[Checkpoint] Loaded RMF, call edge, location edge, and label data")

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

        # Step 3: Map edge lists from index to USR
        # Process call graph
        edge_call_df['i'] = edge_call_df['i'].map(index_to_usr)
        edge_call_df['j'] = edge_call_df['j'].map(index_to_usr)
        
        # Process location graph
        edge_loc_df['i'] = edge_loc_df['i'].map(index_to_usr)
        edge_loc_df['j'] = edge_loc_df['j'].map(index_to_usr)

        # Step 4: Filter out edges with churners
        before_call_edges = len(edge_call_df)
        edge_call_df = edge_call_df[~edge_call_df['i'].isin(churner_set_m1) & ~edge_call_df['j'].isin(churner_set_m1)]
        after_call_edges = len(edge_call_df)
        print(f"[Checkpoint] Removed {before_call_edges - after_call_edges} call edges involving churners")

        before_loc_edges = len(edge_loc_df)
        edge_loc_df = edge_loc_df[~edge_loc_df['i'].isin(churner_set_m1) & ~edge_loc_df['j'].isin(churner_set_m1)]
        after_loc_edges = len(edge_loc_df)
        print(f"[Checkpoint] Removed {before_loc_edges - after_loc_edges} location edges involving churners")

        # Step 5: Map remaining USRs to new 0-based indices
        usr_list_new = node_df['USR'].tolist()
        usr_to_index = {usr: idx for idx, usr in enumerate(usr_list_new)}
        print(f"[Checkpoint] Created mapping for {len(usr_to_index)} remaining users")

        # Step 6: Convert edge lists from USR to index and extract edge weights
        # Process call graph
        mapped_call_i = edge_call_df['i'].map(usr_to_index)
        mapped_call_j = edge_call_df['j'].map(usr_to_index)
        edge_call_weights = edge_call_df['x'].values if 'x' in edge_call_df.columns else np.ones(len(mapped_call_i))
        
        # Process location graph
        mapped_loc_i = edge_loc_df['i'].map(usr_to_index)
        mapped_loc_j = edge_loc_df['j'].map(usr_to_index)
        edge_loc_weights = edge_loc_df['x'].values if 'x' in edge_loc_df.columns else np.ones(len(mapped_loc_i))

        print(f"[Checkpoint] Extracted call edge weights with min={edge_call_weights.min() if len(edge_call_weights) > 0 else 'N/A'}, max={edge_call_weights.max() if len(edge_call_weights) > 0 else 'N/A'}")
        print(f"[Checkpoint] Extracted location edge weights with min={edge_loc_weights.min() if len(edge_loc_weights) > 0 else 'N/A'}, max={edge_loc_weights.max() if len(edge_loc_weights) > 0 else 'N/A'}")

        # Filter out any edges with NA mappings
        valid_call_edges = ~(mapped_call_i.isna() | mapped_call_j.isna())
        mapped_call_i = mapped_call_i[valid_call_edges].values
        mapped_call_j = mapped_call_j[valid_call_edges].values
        edge_call_weights = edge_call_weights[valid_call_edges]

        valid_loc_edges = ~(mapped_loc_i.isna() | mapped_loc_j.isna())
        mapped_loc_i = mapped_loc_i[valid_loc_edges].values
        mapped_loc_j = mapped_loc_j[valid_loc_edges].values
        edge_loc_weights = edge_loc_weights[valid_loc_edges]

        # Create edge indices and weights for undirected graph
        # For call graph
        edge_call_index_0 = torch.tensor([mapped_call_i, mapped_call_j], dtype=torch.long)
        edge_call_index_1 = torch.tensor([mapped_call_j, mapped_call_i], dtype=torch.long)
        edge_call_index = torch.cat([edge_call_index_0, edge_call_index_1], dim=1)
        
        edge_call_weight_0 = torch.tensor(edge_call_weights, dtype=torch.float)
        edge_call_weight_1 = torch.tensor(edge_call_weights, dtype=torch.float)
        edge_call_weight = torch.cat([edge_call_weight_0, edge_call_weight_1], dim=0)
        
        # For location graph
        edge_loc_index_0 = torch.tensor([mapped_loc_i, mapped_loc_j], dtype=torch.long)
        edge_loc_index_1 = torch.tensor([mapped_loc_j, mapped_loc_i], dtype=torch.long)
        edge_loc_index = torch.cat([edge_loc_index_0, edge_loc_index_1], dim=1)
        
        edge_loc_weight_0 = torch.tensor(edge_loc_weights, dtype=torch.float)
        edge_loc_weight_1 = torch.tensor(edge_loc_weights, dtype=torch.float)
        edge_loc_weight = torch.cat([edge_loc_weight_0, edge_loc_weight_1], dim=0)
        
        print(f"[Checkpoint] Created undirected call edge index with shape: {edge_call_index.shape}")
        print(f"[Checkpoint] Created call edge weights with shape: {edge_call_weight.shape}")
        print(f"[Checkpoint] Created undirected location edge index with shape: {edge_loc_index.shape}")
        print(f"[Checkpoint] Created location edge weights with shape: {edge_loc_weight.shape}")

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

        # Step 9: Create and return PyG Data object with edge indices and weights for both graphs
        data = Data(
            x=x, 
            edge_index_call=edge_call_index,
            edge_attr_call=edge_call_weight,
            edge_index_loc=edge_loc_index,
            edge_attr_loc=edge_loc_weight,
            y=y, 
            num_nodes=x.shape[0], 
            num_edges_call=edge_call_index.shape[1],
            num_edges_loc=edge_loc_index.shape[1],
            num_features=x.shape[1]
        )
        print(f"[Checkpoint] Created Data object with {data.num_nodes} nodes, {data.num_edges_call} call edges, {data.num_edges_loc} location edges, {data.num_features} features")

        return data

    @staticmethod
    def get_class_distribution(data):
        """Returns class distribution for monitoring"""
        y = data.y
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        print(f"[Checkpoint] Class distribution - Positives: {num_pos}, Negatives: {num_neg}")
        return num_pos, num_neg

class EnhancedDualGCN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_channels, num_layers, num_nodes, dropout_rate=Config.DROPOUT_RATE):
        super().__init__()
        print(f"[Checkpoint] Initializing EnhancedDualGCN with {input_dim} input features, {embedding_dim} embedding dim, {hidden_channels} hidden channels, {num_layers} layers")
        
        # Initialize weights with smaller values to prevent exploding gradients
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.1)
        
        self.feature_transform = nn.Linear(input_dim, embedding_dim)
        nn.init.xavier_uniform_(self.feature_transform.weight, gain=0.1)
        
        self.combine = nn.Linear(embedding_dim * 2, hidden_channels)
        nn.init.xavier_uniform_(self.combine.weight, gain=0.1)
        
        # Two sets of GCN layers for call and location graphs
        self.call_convs = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        
        for i in range(num_layers):
            call_conv = GCNConv(hidden_channels, hidden_channels)
            loc_conv = GCNConv(hidden_channels, hidden_channels)
            # Initialize GCN layers with appropriate initialization
            nn.init.xavier_uniform_(call_conv.lin.weight, gain=0.1)
            nn.init.xavier_uniform_(loc_conv.lin.weight, gain=0.1)
            self.call_convs.append(call_conv)
            self.loc_convs.append(loc_conv)
            
        # Layer to combine outputs from both graphs
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        nn.init.xavier_uniform_(self.fusion.weight, gain=0.1)
        
        self.lin = nn.Linear(hidden_channels, 1)
        nn.init.xavier_uniform_(self.lin.weight, gain=0.1)
        
        self.dropout_rate = dropout_rate
        
        # Add BatchNorm to help with training stability
        self.batch_norm_call = nn.BatchNorm1d(hidden_channels)
        self.batch_norm_loc = nn.BatchNorm1d(hidden_channels)
        self.batch_norm_fusion = nn.BatchNorm1d(hidden_channels)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Checkpoint] Total parameters: {total_params}")

    def forward(self, x, edge_index_call, edge_weight_call, edge_index_loc, edge_weight_loc):
        # Input validation
        if torch.isnan(x).any():
            print("[Checkpoint] WARNING: NaN detected in input features")
            x = torch.nan_to_num(x, nan=0.0)
            
        node_indices = torch.arange(x.size(0), device=x.device)
        node_emb = self.embedding(node_indices)
        feature_emb = self.feature_transform(x)
        
        combined = torch.cat([node_emb, feature_emb], dim=1)
        h = F.relu(self.combine(combined))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        # Process call graph
        h_call = h.clone()
        for conv in self.call_convs:
            h_call_new = conv(h_call, edge_index_call, edge_weight_call)
            h_call_new = F.relu(h_call_new)
            h_call_new = self.batch_norm_call(h_call_new)
            h_call_new = F.dropout(h_call_new, p=self.dropout_rate, training=self.training)
            
            # Add residual connection
            if h_call.shape == h_call_new.shape:
                h_call = h_call_new + h_call
            else:
                h_call = h_call_new
        
        # Process location graph
        h_loc = h.clone()
        for conv in self.loc_convs:
            h_loc_new = conv(h_loc, edge_index_loc, edge_weight_loc)
            h_loc_new = F.relu(h_loc_new)
            h_loc_new = self.batch_norm_loc(h_loc_new)
            h_loc_new = F.dropout(h_loc_new, p=self.dropout_rate, training=self.training)
            
            # Add residual connection
            if h_loc.shape == h_loc_new.shape:
                h_loc = h_loc_new + h_loc
            else:
                h_loc = h_loc_new
        
        # Fusion of call and location embeddings
        h_fusion = torch.cat([h_call, h_loc], dim=1)
        h_fusion = self.fusion(h_fusion)
        h_fusion = F.relu(h_fusion)
        h_fusion = self.batch_norm_fusion(h_fusion)
        h_fusion = F.dropout(h_fusion, p=self.dropout_rate, training=self.training)
        
        # Final prediction
        out = self.lin(h_fusion)
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
        edge_index_call = data.edge_index_call.to(self.device)
        edge_weight_call = data.edge_attr_call.to(self.device)
        edge_index_loc = data.edge_index_loc.to(self.device)
        edge_weight_loc = data.edge_attr_loc.to(self.device)
        y = data.y.float().to(self.device)
        
        # Normalize edge weights to prevent numerical issues
        if edge_weight_call.max() > 1000:
            print("[Checkpoint] Normalizing large call edge weights")
            edge_weight_call = edge_weight_call / edge_weight_call.max()
            
        if edge_weight_loc.max() > 1000:
            print("[Checkpoint] Normalizing large location edge weights")
            edge_weight_loc = edge_weight_loc / edge_weight_loc.max()
        
        out = self.model(x, edge_index_call, edge_weight_call, edge_index_loc, edge_weight_loc)
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
            edge_index_call = data.edge_index_call.to(self.device)
            edge_weight_call = data.edge_attr_call.to(self.device)
            edge_index_loc = data.edge_index_loc.to(self.device)
            edge_weight_loc = data.edge_attr_loc.to(self.device)
            y = data.y.float().to(self.device)
            
            # Normalize edge weights to prevent numerical issues
            if edge_weight_call.max() > 1000:
                edge_weight_call = edge_weight_call / edge_weight_call.max()
                
            if edge_weight_loc.max() > 1000:
                edge_weight_loc = edge_weight_loc / edge_weight_loc.max()
            
            try:
                out = self.model(x, edge_index_call, edge_weight_call, edge_index_loc, edge_weight_loc)
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
            edge_call_path=Config.TRAIN_EDGE_CALL, 
            edge_loc_path=Config.TRAIN_EDGE_LOC,
            label_path=Config.TRAIN_LABEL,
            rmf_path=Config.TRAIN_RMF,
            churner_set_m1=self.churned_users
        )
        GraphDataProcessor.get_class_distribution(self.data_train)
        
        print("[Checkpoint] Loading validation data with RMF features")
        self.data_val = GraphDataProcessor.remove_nodes_and_create_data(
            edge_call_path=Config.VAL_EDGE_CALL,
            edge_loc_path=Config.VAL_EDGE_LOC,
            label_path=Config.VAL_LABEL,
            rmf_path=Config.VAL_RMF,
            churner_set_m1=self.churned_users
        )

        print("[Checkpoint] Loading test data with RMF features")
        self.data_test = GraphDataProcessor.remove_nodes_and_create_data(
            edge_call_path=Config.TEST_EDGE_CALL,
            edge_loc_path=Config.TEST_EDGE_LOC,
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
                                model = EnhancedDualGCN(
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