import matplotlib.pyplot as plt

def plot_curves(train_losses, val_losses, save_path):
    """
    绘制并保存训练和验证损失曲线
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")