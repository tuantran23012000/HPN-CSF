import torch
from torch import nn, sigmoid
import torch.nn.functional as F
device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
class Toy_Hypernetwork_2d(nn.Module):
  def __init__(self, ray_hidden_dim=100, out_dim=10, n_hidden=1, n_tasks=2):
      super().__init__()
      self.n_hidden = n_hidden
      self.n_tasks = n_tasks
      '''
            chebyshev example 7.1
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.Sigmoid()
      '''
      '''
            linear example 7.1
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.Sigmoid()
      '''
      '''
            utility example 7.1
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.Sigmoid()
      '''
      '''
            linear example 7.2
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            chebyshev example 7.2
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            utility example 7.2
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            KL example 7.2
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            Cosine example 7.2
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            Cauchy example 7.2
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            KL example 7.1
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            cosine example 7.1
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      '''
            cauchy example 7.1
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
      '''
      self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(ray_hidden_dim, ray_hidden_dim),
            # nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
  def shared_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('fc3')]
  def forward(self, ray):
      features = self.ray_mlp(ray)
      return features.unsqueeze(0)
class Toy_Hypernetwork_3d(nn.Module):
  def __init__(self, ray_hidden_dim=100, out_dim=10, n_hidden=1, n_tasks=2):
      super().__init__()
      self.n_hidden = n_hidden
      self.n_tasks = n_tasks
      '''
            convex problem monotonic utility -3D
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(ray_hidden_dim, out_dim),
            #nn.ReLU(inplace=True),
            #nn.Linear(ray_hidden_dim, ray_hidden_dim),
            #LearnedSiLU(ray_hidden_dim)
            nn.Softmax(dim=0)
      '''
      '''
            convex problem monotonic linear -3D
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            #nn.ReLU(inplace=True),
            # nn.Linear(ray_hidden_dim, ray_hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(ray_hidden_dim, ray_hidden_dim),
            # nn.ReLU(inplace=True),

            nn.Linear(ray_hidden_dim, out_dim),
            #nn.ReLU(inplace=True),
            #nn.Linear(ray_hidden_dim, ray_hidden_dim),
            #LearnedSiLU(ray_hidden_dim)
            nn.Softmax(dim=0)
      '''
      '''
            cheby - 3D
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.Softmax(dim=0)
      '''
      '''
            KL problem - 3D
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.Softmax(dim=0)
      '''
      '''
            Cosine problem - 3D 
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.Softmax(dim=0)
      '''
      '''   
            Cauchy-Schwarz problem - 3D
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(ray_hidden_dim, out_dim),
            nn.Softmax(dim=0)
      '''
      self.ray_mlp = nn.Sequential(
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
            nn.Softmax(dim=0)
        )
  def forward(self, ray):
      features = self.ray_mlp(ray)
      return features.unsqueeze(0)
