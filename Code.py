import torch
import torch.nn as nn
from torchtext.legacy import data
from torchtext.legacy import datasets

# 定义PCA模型
class DualPathCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs_avg = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.convs_max = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        avg_pooled = [torch.mean(F.relu(conv(embedded)).squeeze(3), dim=2) for conv in self.convs_avg]
        max_pooled = [torch.max(F.relu(conv(embedded)).squeeze(3), dim=2)[0] for conv in self.convs_max]
        pooled = avg_pooled + max_pooled
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# 超参数设置
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5

# 实例化模型、优化器和损失函数
model = DualPathCNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 加载训练好的模型
model.load_state_dict(torch.load('dualpathcnn.pth'))
model.eval()

# 测试数据预处理
def preprocess_test_data(text):
    tokenized = [tok.text for tok in spacy_en.tokenizer(text)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).unsqueeze(1)
    return tensor

# 测试函数
def test(model, test_iterator):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in test_iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            test_loss += loss.item()
            test_acc += acc.item()
    return test_loss / len(test_iterator), test_acc / len(test_iterator)

# 加载测试数据并运行测试函数
test_data = "This is a test sentence."
test_tensor = preprocess_test_data(test_data)
test_iterator = data.Iterator(test_dataset, batch_size=1, train=False, sort=False)
test_loss, test_acc = test(model, test_iterator)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc*100:.2f}%')
