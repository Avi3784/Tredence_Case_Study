import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # standard weights and bias initialization (kaiming for relu)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        # gate scores (initialized to 1.5 for a strong starting sigmoid)
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.constant_(self.gate_scores, 1.5)

    def forward(self, x):
        # 3. apply sigmoid to get gates between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # 4. multiply weights by gates to prune them
        pruned_weights = self.weight * gates
        
        # 5. normal linear operation
        return F.linear(x, pruned_weights, self.bias)

class PruningMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x): 
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def get_all_gates(self):
        # dynamically collect all gate scores from the network
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(torch.sigmoid(module.gate_scores).view(-1))
        return torch.cat(gates)

def train(model, loader, optimizer, lam, device):
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # forward pass
        logits = model(x)
        
        # 1. normal classification loss
        ce_loss = F.cross_entropy(logits, y)
        
        # 2. sparsity loss (L1 sum of all gates)
        gates = model.get_all_gates()
        sparsity_loss = gates.abs().sum()
        
        # 3. total loss
        total_loss = ce_loss + (lam * sparsity_loss)
        
        # backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            predictions = logits.argmax(dim=1)
            
            correct += (predictions == y).sum().item()
            total += len(y)
            
    accuracy = 100.0 * correct / total
    return accuracy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    
    # load data simply
    transform = T.Compose([T.ToTensor()])
    train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    results = []
    best_model = None
    best_score = -1
    
    # test 3 different lambda values
    for lam in [0.0001, 0.001, 0.01]:
        print(f"\n--- testing lambda: {lam} ---")
        
        model = PruningMLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # train for 15 epochs
        for epoch in range(1, 16):
            train(model, train_loader, optimizer, lam, device)
            accuracy = evaluate(model, test_loader, device)
            
            # calculate sparsity level (% of gates under 0.01)
            gates = model.get_all_gates()
            sparsity = 100.0 * (gates < 0.01).sum().item() / len(gates)
            
            print(f"epoch {epoch} | accuracy: {accuracy:.2f}% | sparsity: {sparsity:.2f}%")
            
        results.append((lam, accuracy, sparsity))
        
        # save the best model to make the plot later
        score = accuracy + sparsity
        if score > best_score:
            best_score = score
            best_model = model

    # save plot
    final_gates = best_model.get_all_gates().detach().cpu().numpy()
    plt.hist(final_gates, bins=50, color="blue", log=True)
    plt.title("gate distribution")
    plt.savefig("gate_distribution.png")

    # save report
    report = "# self pruning network report\n\n"
    report += "## l1 penalty explanation\n"
    report += "the l1 norm minimizes values towards zero. since sigmoid outputs are always positive, "
    report += "minimizing the l1 norm forces the gates to become exactly zero, which deletes the weights.\n\n"
    
    report += "## experimental results\n"
    report += "| Lambda | Test Accuracy | Sparsity Level (%) |\n"
    report += "|---|---|---|\n"
    for lam, acc, spar in results:
        report += f"| {lam} | {acc:.2f}% | {spar:.2f}% |\n"
        
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()
