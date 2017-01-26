local TaskLossFactory = torch.class("TaskLossFactory")

function TaskLossFactory:__init(loss_type)
   self.loss_type = loss_type
   self.M = torch.Tensor({{1, 1, 1, 1, 1, 1, 0},
                          {1, 1, 0, 0, 0, 0, 0},
                          {1, 0, 1, 1, 0, 1, 1},
                          {1, 0, 0, 1, 1, 1, 1},
                          {0, 1, 0, 0, 1, 1, 1},
                          {1, 1, 0, 1, 1, 0, 1},
                          {1, 1, 1, 1, 1, 0, 1},
                          {1, 0, 0, 0, 1, 1, 0},
                          {1, 1, 1, 1, 1, 1, 1},
                          {1, 1, 0, 1, 1, 1, 1}})
    if loss_type == "dist_num" then
      self.cost = {eval = self.dist_num}
    elseif loss_type == 'dist_segments' then
      self.cost = {eval = self.dist_segments}
    end
end

function TaskLossFactory:get_loss(y, y_hat)
  if (torch.isTensor(y) and not torch.isTensor(y_hat)) or (not torch.isTensor(y) and torch.isTensor(y_hat)) then
    print('ERROR with loss function.')
    return -1
  else
    return self.cost.eval(self, y, y_hat)
  end
end

function TaskLossFactory:cuda()
   self.M = self.M:cuda()
end

-- this loss function is the numerical distance between the digits multiply by two
-- L(y, y_hat) = 2 * ||y - y_hat||
function TaskLossFactory.dist_num(self, y, y_hat)
  y = y:long()
  y_hat = y_hat:long()
  return (1 - torch.eq(y, y_hat)):float()
end

-- this loss function is the seven segments distance between the digits
function TaskLossFactory.dist_segments(self, y, y_hat)
  if not torch.isTensor(y) then
      return torch.abs(self.M[y] - self.M[y_hat])
  else
    return torch.sum(torch.abs(self.M:index(1, y) - self.M:index(1, y_hat)), 2):t()[1]
  end
end

return TaskLossFactory