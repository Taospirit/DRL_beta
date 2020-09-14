from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('./log')
for i in range(1000):
    writer.add_scalar('test', i**2, global_step=i)
writer.close()