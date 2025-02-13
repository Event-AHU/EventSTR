import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, num_patterns, pattern_dim, feature_dim,args=None):
        super(Memory, self).__init__()
        self.num_patterns = num_patterns
        self.pattern_dim = pattern_dim
        self.feature_dim = feature_dim
        self.args = args

        self.patterns = nn.Parameter(torch.randn(num_patterns, pattern_dim), requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.encoder = nn.Linear(feature_dim, pattern_dim)
        self.decoder = nn.Linear(pattern_dim, feature_dim)


    def forward(self, input_features):

        # B, L, D -> B*L, D
        B, L, D = input_features.size()  #3,14,1024  2,256,1408
        input_features = input_features.reshape(-1, D) #torch.Size([42, 1024]) 512,1408
        # 将输入特征编码为模式维度
        encode_features = self.encoder(input_features) #torch.Size([42, 128])  512,128
        # input_features = input_features.view(-1, input_features.size(-1))
        # 计算输入特征与模式池的相似度  torch.Size([42, 128])
        similarities = F.cosine_similarity(encode_features.unsqueeze(1), self.patterns.unsqueeze(0), dim=2)
        # 找到最相似的模式
        top_k = 128  # 选择前K个相似模式
        _, indices = torch.topk(similarities, top_k, dim=1)
        # 从模式池中获取相似的模式
        closest_patterns = self.patterns[indices]
        # 解码获得增强的特征
        closest_patterns = self.decoder(closest_patterns) #torch.Size([512, 32, 1408])
        # 特征增强
        enhanced_features = input_features + self.alpha[0] * torch.mean(closest_patterns, dim=1) #torch.Size([512, 1408])
        # B*L, D -> B, L, D
        enhanced_features = enhanced_features.reshape(B, L, D) #torch.Size([2, 256, 1408])
        
        # return enhanced_features,0
        return enhanced_features

