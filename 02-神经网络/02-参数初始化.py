import torch.nn as nn
# 参数初始化
linear = nn.Linear(3,5)
print("均匀分布",nn.init.uniform_(linear.weight))
print("正太分布",nn.init.normal_(linear.weight))
print("全0初始化",nn.init.zeros_(linear.weight))
print("全1初始化",nn.init.ones_(linear.weight))
print("指定值初始化",nn.init.constant_(linear.weight,100))
# bias 正常全0或全1初始化
print("全0初始化",nn.init.zeros_(linear.bias))
print("凯明均匀初始化",nn.init.kaiming_normal_(linear.weight))
print("凯明正太分布",nn.init.kaiming_uniform_(linear.weight))
print("xavier均匀分布",nn.init.xavier_normal_(linear.weight))
print("xavier正太分布",nn.init.xavier_uniform_(linear.weight))