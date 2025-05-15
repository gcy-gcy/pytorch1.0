from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')  # 实例化一个SummaryWriter为writer，并指定event的保存路径为logs
for i in range(100):
    writer.add_scalar('y=3x', 3 * i, i)
writer.close()  # 最后还需要将这个writer关闭
