import torch
import torch.nn as nn

class SCL(nn.Module):
  '''compute Cycle-Contrast loss'''
  def __init__(self):
    super(SCL, self).__init__()

  def forward(self, style_fea, other_fea, t = 1):

      # style_fea: b * dim
      # other_fea: b * 3 * num * dim
      batch_size = style_fea.shape[0]
      num_seg = other_fea.shape[2]
      dim = style_fea.shape[1]
      style_fea = style_fea.reshape(batch_size, 1, 1, dim)
      style_fea = style_fea.expand(batch_size, 3, num_seg, dim)
      asim = torch.cosine_similarity(style_fea, other_fea, dim=3)
      asim = asim.reshape(batch_size, 3 * num_seg)
      asim = torch.softmax(asim, 1)
      asim = asim.reshape(batch_size, 3, num_seg, 1)
      seg_fea_wei = other_fea * asim
      seg_fea_wei = seg_fea_wei.reshape(batch_size, 3 * num_seg, dim)
      z_wei = torch.sum(seg_fea_wei, 1)
      z_wei = z_wei.reshape(batch_size, 1, 1, dim)
      z_wei_new = z_wei.expand(batch_size, 3, num_seg, dim)
      sim_floss = torch.cosine_similarity(z_wei_new, other_fea, dim=3)
      sim_floss = sim_floss / t
      sim_floss = sim_floss.reshape(batch_size, 3 * num_seg)
      sim_floss = torch.softmax(sim_floss, 1)
      sim_floss = sim_floss.reshape(batch_size, 3, num_seg)
      floss = -torch.log(sim_floss)
      floss = floss[:, 0, :]
      loss = torch.mean(floss)

      return  loss


if __name__ == "__main__":
    style_fea = torch.randn(8, 128)
    other_fea = torch.randn(8, 3, 26, 128)
    scl = SCL()
    loss = scl(style_fea, other_fea, 0.1)
    print(loss)