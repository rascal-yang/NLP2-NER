import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import BertForTokenClassification

from torchcrf import CRF

class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(BertAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=0.1)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Prepare attention mask
        if attention_mask is not None:
            # Ensure the attention mask has shape [seq_len, seq_len]
            attention_mask = attention_mask.view(hidden_states.size(0), 1, -1)  # [batch_size, 1, seq_len]
            attention_mask = attention_mask.repeat(1, hidden_states.size(1), 1)  # [batch_size, seq_len, seq_len]
            # Repeat attention_mask num_attention_heads times on the 0th dimension
            attention_mask = attention_mask.repeat_interleave(self.num_attention_heads, dim=0)

        # multihead self-attention
        attention_output, _ = self.multihead_attention(mixed_query_layer, mixed_key_layer, mixed_value_layer, 
                                                       attn_mask=attention_mask.bool(), need_weights=False)

        attention_output = self.dropout(attention_output)

        return attention_output



class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(BertSelfAttention, self).__init__()
        # 注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 每个注意力头的大小
        self.attention_head_size = int(hidden_size / num_attention_heads)
        # 所有注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建 query、key 和 value 的线性变换层
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=True)

        # Dropout 层，用于随机失活
        self.dropout = nn.Dropout(0.1, inplace=False)

    def transpose_for_scores(self, x):
        # 将线性变换后的结果重新形状，以便进行注意力计算
        # 将 x 的最后一个维度拆分成两个维度，分别是注意力头的数量和每个注意力头的大小。
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # 使用 query、key 和 value 线性变换得到 mixed_query_layer、mixed_key_layer 和 mixed_value_layer
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # print(mixed_value_layer.shape) torch.Size([32, 128, 768])

        # 将 mixed_query_layer、mixed_key_layer 和 mixed_value_layer 转换为多头注意力的形式
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print(query_layer.shape) torch.Size([32, 8, 128, 96])

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print(attention_scores.shape) torch.Size([32, 8, 128, 128])
        # print(attention_scores)

        # 对注意力掩码进行处理，将值为0的位置处的注意力分数置为一个很小的负数
        attention_mask = (1 - attention_mask) * -10000.0
        # print(attention_mask.shape)
        # torch.Size([32, 8, 128, 128])
        # 扩展注意力掩码的维度
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores + extended_attention_mask

        # 对注意力分数进行 softmax 归一化，并进行随机失活
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文表示
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # (batch_size, sequence_length, hidden_size)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super(BertSelfOutput, self).__init__()
        # 线性变换层 线性变换有助于模型学习输入的复杂特征和模式
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12, elementwise_affine=True)
        # Dropout 层，用于随机失活
        self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, hidden_states, input_tensor):
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 随机失活
        hidden_states = self.dropout(hidden_states)
        # 残差连接和 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(BertIntermediate, self).__init__()
        '''将输入的隐藏状态通过一个线性变换扩展为更高维度的特征表示，然后应用 GELU 激活函数'''
        # 线性变换层，将隐藏状态的维度扩展为hidden_size * 4
        self.dense = nn.Linear(hidden_size, hidden_size * 4, bias=True)
        # 激活函数，GELU激活函数
        self.gelu = nn.GELU()

    def forward(self, hidden_states):
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # GELU激活函数
        hidden_states = self.gelu(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, hidden_size):
        super(BertOutput, self).__init__()
        # 线性变换层，将hidden_size * 4的维度转换为hidden_size
        self.dense = nn.Linear(hidden_size * 4, hidden_size, bias=True)
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12, elementwise_affine=True)
        # Dropout 层，用于随机失活
        self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, hidden_states, input_tensor):
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 随机失活
        hidden_states = self.dropout(hidden_states)
        # 残差连接和 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(BertLayer, self).__init__()
        # BertSelfAttention
        self.attention = BertSelfAttention(hidden_size, num_attention_heads)
        self.attention_output = BertSelfOutput(hidden_size)
        # BertIntermediate
        self.intermediate = BertIntermediate(hidden_size)
        self.output = BertOutput(hidden_size)

    def forward(self, hidden_states, attention_mask):
        # 自注意力机制
        attention_output = self.attention(hidden_states, attention_mask)
        # 自注意力输出 经过了残差和归一化
        attention_output = self.attention_output(attention_output, hidden_states)
        # Feedforward部分
        intermediate_output = self.intermediate(attention_output)
        # Feedforward输出 经过了残差和归一化
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads):
        super(BertEncoder, self).__init__()
        # 创建多层 BertLayer
        layer = BertLayer(hidden_size, num_attention_heads)
        self.layer = nn.ModuleList([layer for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        # 遍历所有 BertLayer 层
        for layer_module in self.layer:
            # 前向传播，得到每一层的隐藏状态
            hidden_states = layer_module(hidden_states, attention_mask)
            # 将每一层的隐藏状态保存起来
            all_encoder_layers.append(hidden_states)
        # 返回所有层的隐藏状态
        return all_encoder_layers


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings):
        # vocab_size 词汇表大小 21128
        # hidden_size 隐藏层大小 768
        # max_position_embeddings 序列的最大长度 512
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        # 对每个隐藏单元应用一个可学习的缩放和偏置
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12, elementwise_affine=True)
        # 不原地执行 dropout 操作
        self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, input_ids, token_type_ids=None):
        # 获取输入序列的长度
        seq_length = input_ids.size(-1)
        # 创建位置编码张量，其值为 [0, 1, 2, ..., seq_length-1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # 将位置编码张量扩展成与输入序列相同的维度
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 使用词嵌入层将输入的 token 序列转换为词嵌入向量
        words_embeddings = self.word_embeddings(input_ids)
        # 使用位置嵌入层将位置编码序列转换为位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        # 使用标记类型嵌入层将标记类型序列转换为标记类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入向量、位置嵌入向量和标记类型嵌入向量相加得到最终的嵌入向量
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        # 使用 LayerNorm 对嵌入向量进行归一化
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, num_hidden_layers, num_attention_heads):
        super(BertModel, self).__init__()
        # self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.embeddings = BertForTokenClassification.from_pretrained('./bert_model').bert.embeddings
        self.encoder = BertEncoder(num_hidden_layers, hidden_size, num_attention_heads)
        
        # 冻结预训练的词嵌入层参数
        for param in self.embeddings.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, attention_mask)
        sequence_output = encoded_layers[-1]
        return sequence_output


class BertForClassification(nn.Module):
    def __init__(self, vocab_size, max_position_embeddings, \
            num_hidden_layers, hidden_size, num_attention_heads, num_labels):
        super(BertForClassification, self).__init__()
        self.bert = BertModel(vocab_size, hidden_size, max_position_embeddings, num_hidden_layers, num_attention_heads)
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.classifier = nn.Linear(hidden_size, num_labels, bias=True)
        self.loss = 0.0

    def forward(self, input_ids, attention_mask, labels=None):
        sequence_output = self.bert(input_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
            # 应用注意力掩码，过滤掉填充部分
            masked_loss = torch.masked_select(loss, attention_mask.view(-1).bool())
            # 计算平均损失
            average_loss = torch.mean(masked_loss)
            self.loss = average_loss

        return logits
    
class BertModelWithCRF(nn.Module):
    def __init__(self, vocab_size, max_position_embeddings, \
            num_hidden_layers, hidden_size, num_attention_heads, num_labels):
        super(BertModelWithCRF, self).__init__()
        
        self.bert = BertModel(vocab_size, hidden_size, max_position_embeddings, num_hidden_layers, num_attention_heads)
        
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.classifier = nn.Linear(hidden_size, num_labels, bias=True)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True) # 设置为True 表示输入的第一个维度是batch_size (batch_size, sequence_length, num_labels)
        self.loss = 0.0

    def forward(self, input_ids, attention_mask, labels=None):

        sequence_output = self.bert(input_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, sequence_length, num_labels)
        
        # 如果labels不为None 则计算CRF的log-likelihood作为损失
        if labels is not None:
            # 计算CRF的log-likelihood作为损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            self.loss = -log_likelihood

        # 使用CRF解码预测序列
        predictions = self.crf.decode(logits, mask=attention_mask.byte())
        # 创建一个形状为 [batch_size, sequence_length] 的张量
        prediction_tensor = torch.zeros(input_ids.size(0), input_ids.size(1), dtype=torch.long, device=input_ids.device)

        # 将解码后的预测序列填充到张量中
        for i, prediction in enumerate(predictions):
            prediction_tensor[i, :len(prediction)] = torch.tensor(prediction, device=input_ids.device)

        return prediction_tensor.float()

if __name__ == "__main__":
    # 定义模型参数
    vocab_size = 21128  # 词汇表大小
    hidden_size = 768  # 隐藏层大小
    max_position_embeddings = 512  # 序列的最大长度
    num_hidden_layers = 12  # Transformer 层的数量
    num_attention_heads = 8  # 注意力头的数量
    num_labels = 2  # 标签数量

    # 初始化BERT分类模型对象
    bert_classifier = BertForClassification(vocab_size, max_position_embeddings,\
            num_hidden_layers, hidden_size, num_attention_heads, num_labels)



