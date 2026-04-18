import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, train_ds, val_ds, epochs=50, lr=1e-3, batch=64, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch)

    history = {'train_loss': [], 'val_loss': []}
    best_val, best_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        # Validation
        model.eval()
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_total += criterion(model(X), y).item()

        train_loss = total / len(train_loader)
        val_loss = val_total / len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, history