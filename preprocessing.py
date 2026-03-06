import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Load data
train_df = pd.read_csv("data/KDDTrain+.txt", names=col_names)
test_df  = pd.read_csv("data/KDDTest+.txt",  names=col_names)

# Encode categorical columns
for col in ["protocol_type", "service", "flag"]:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])

# Separate labels
train_labels = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
test_labels  = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

# Drop label and difficulty columns
train_df = train_df.drop(columns=["label", "difficulty"])
test_df  = test_df.drop(columns=["label", "difficulty"])

# Normalize to [0, 1]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled  = scaler.transform(test_df)

# PCA to reduce to 8 features
pca = PCA(n_components=8)
train_pca = pca.fit_transform(train_scaled)
test_pca  = pca.transform(test_scaled)

# Scale to [0, pi] for quantum angle encoding
train_pca = train_pca * np.pi
test_pca  = test_pca  * np.pi

# Save
np.save("data/train_data.npy",   train_pca)
np.save("data/train_labels.npy", train_labels.values)
np.save("data/test_data.npy",    test_pca)
np.save("data/test_labels.npy",  test_labels.values)

print("Preprocessing done!")
print(f"Train shape: {train_pca.shape}")
print(f"Test shape:  {test_pca.shape}")
print(f"Train - Normal: {(train_labels==0).sum()}, Anomaly: {(train_labels==1).sum()}")
print(f"Test  - Normal: {(test_labels==0).sum()},  Anomaly: {(test_labels==1).sum()}")