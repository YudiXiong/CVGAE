"""
Author: Yudi Xiong
Google Scholar: https://scholar.google.com/citations?user=LY4PK9EAAAAJ
ORCID: https://orcid.org/0009-0001-3005-8225
Date: April, 2024
Example command to run:
python camouflaged_attribute_embedding.py --csv_file_path='user_camouflaged_embeddings_gender.csv'
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Learn camouflaged_attribute_embedding.")
    parser.add_argument('--csv_file_path', nargs='?', default='user_camouflaged_embeddings_gender.csv',
                        help='The path where the camouflaged embeddings file needs to be saved.')
    return parser.parse_args()
args = parse_args()

np.random.seed(2024)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.emb = nn.Embedding(2068, input_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, indices):
        x = self.emb(indices)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

n_users = 2068
embedding_dim = 128
hidden_dim = 128
output_dim = 8  # 2 or 8


file_path = '../Data/QBarticle_QBvideo/new_reindex.txt'

df = pd.read_csv(file_path, sep='\t', header=None)

df.columns = ['User_ID', 'Item_ID', 'Rating', 'Gender', 'Age']

user_gender_matrix = df[['User_ID', 'Gender']].drop_duplicates().reset_index(drop=True)

user_gender_matrix['Gender'] = user_gender_matrix['Gender'] - 1

user_age_matrix = df[['User_ID', 'Age']].drop_duplicates().reset_index(drop=True)
Y_gender = user_gender_matrix.iloc[:, 1].tolist()
Y_age = user_age_matrix.iloc[:, 1].tolist()

# probability p
replace_prob = 0.8




Y_gender_camouflaged = np.array([np.random.choice([j for j in range(2) if j != Y_gender[i]])
                           if np.random.rand() <= replace_prob else Y_gender[i]
                           for i in range(len(Y_gender))])

Y_age_camouflaged = np.array([np.random.choice([j for j in range(8) if j != Y_age[i]])
                           if np.random.rand() <= replace_prob else Y_age[i]
                           for i in range(len(Y_age))])

model = MLP(embedding_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()


inputs = torch.tensor([i for i in range(n_users)], dtype=torch.long)
camouflaged_label = torch.LongTensor(Y_age_camouflaged)


epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, camouflaged_label)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


np.savetxt(args.csv_file_path, model.emb.weight.data.numpy(), delimiter=',')




