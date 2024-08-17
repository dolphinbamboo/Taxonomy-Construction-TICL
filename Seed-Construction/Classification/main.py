import torch
from datasets import load_dataset
import torch.utils.data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#构造一个TensorBoard对象，用于向logs文件写入accuracy和loss，comment用于记录当前模型的学习率和batch
writer = SummaryWriter("/home/chaizhihua/LearningSpace/Predict_IsA_bert/logs",comment="LR_0.1_BATCH_16")

lr = 5e-6
epoch_number = 1
Batch_Size = 2

#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self,split) -> None:
        super().__init__()
        #采用huggingface中的load_dataset方法加载本地数据
        self.dataset = load_dataset(
        "csv",#数据格式
        data_files={"train":"/home/chaizhihua/LearningSpace/Predict_IsA_bert/temp.csv",
        "test":"/home/chaizhihua/LearningSpace/Predict_IsA_bert/data_test.csv"},#字典形式，分别指定训练数据和测试数据的位置
        split=split#split在构造此类的实例时指定。split="train"表示取训练集
        )
    
    def __len__(self):
        #返回数据集长度
        return len(self.dataset)
    
    def __getitem__(self,i):
        #此函数返回第i个数据
        sentence = self.dataset[i]["sentence"]#取第i个数据的句子，"sentence"和"label"是我们csv数据的第一行，为数据的属性。
        label = self.dataset[i]["label"]#取第i个数据的标签
        
        return sentence,label

#分别构造训练集和测试集
test_dataset = Dataset("test")
train_dataset = Dataset("train")

#使用transformers的分词器
from transformers import BertTokenizer
#构造分词器
token=BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")


def collate_fn(data):
    '''
    定义数据处理函数，此函数将作为DataLoader的参数。
    '''
    #将data中的句子和label分别放入两个列表中
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(#对句子进行编码
        batch_text_or_text_pairs = sents,
        truncation = True,#句子长度超过max_length的句子要进行截断
        padding = "max_length",#句子长度不足max_length的句子要进行填充
        max_length = 60,
        return_tensors = "pt",#pt表示，编码后的句子用pytorch格式返回，可取值tf,pt,np,默认为返回list  tf:tensorflow   pt:pytorch  np:numpy
        return_length = True#返回数据个数
    )

    #对句子进行编码后，会生成以下三个数据
    input_ids = data["input_ids"]#句子的编码
    attention_mask = data['attention_mask']#pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']#第一个句子和特殊符号的位置是0,第二个句子的位置是1

    #将labels转换为torch的格式
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels

#构造dataloader
loader_train = torch.utils.data.DataLoader(dataset=train_dataset,#选择数据集
                                     batch_size=Batch_Size,
                                     collate_fn=collate_fn,#选择数据处理函数
                                     shuffle=True,#打乱数据的顺序
                                     drop_last=True#当总数据量不是batchsize的整数倍时，丢弃作为余数的那部分数据
                                     )

loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=Batch_Size,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

#导入预训练的bert模型，并将之命名为pretrained_Layer
from transformers import BertModel
pretrained_Layer = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

class Model(torch.nn.Module):
    #定义模型
    def __init__(self):
        super().__init__()
        #首先是bert层
        self.bert = pretrained_Layer
        #然后是线性层，768是bert层输出的维度
        self.Linear = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        
        out = self.Linear(out.last_hidden_state[:, 0])
        return out

model = Model()
#将模型放入GPU训练
model = model.cuda()

def test(num):
    '''
    定义测试函数，在每轮epoch结束后进行测试
    '''
    #进入评估模式，不再计算梯度
    model.eval()
    correct = 0
    total = 0

    for input_ids, attention_mask, token_type_ids,labels in loader_test:
        #将数据放到显卡中运算
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        labels = labels.cuda()

        with torch.no_grad():#不计算梯度
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

            predict = out.argmax(dim=1)
            #累计正确个数
            correct += (predict == labels).sum().item()
            total += len(labels)

    print(f"epoch:{num},accuracy:{correct / total}")

#训练
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()#注意，使用CrossEntropyLoss()时，Model中不能放softmax层
criterion = criterion.cuda()#损失函数也要放到显卡中计算


step = 0#用于TensorBoard对象记录训练次数
for epoch in range(epoch_number):
    model.train()#开启训练模式
    loop = tqdm(loader_train, total =len(loader_train))#使用tqdm是为了显示程序运行的进度条
    for input_ids, attention_mask, token_type_ids,labels in loop:

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        labels = labels.cuda()

        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        
        loss = criterion(out, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        predictions = out.argmax(dim=1)
        running_train_acc = (predictions == labels).sum().item()/ len(labels)
        
        #TensorBoard记录训练数据
        writer.add_scalar('Training loss',loss,global_step=step)
        writer.add_scalar('Training accuracy',running_train_acc,global_step = step)
        step+=1

        #更新进度条信息
        loop.set_description(f'Epoch [{epoch}/{epoch_number}]')
        loop.set_postfix(loss = loss.item(),acc = running_train_acc)
    
    test(epoch)

    torch.save(model,f"/home/chaizhihua/LearningSpace/Predict_IsA_bert/My_Bert_epoch{epoch}.pth")

writer.close()