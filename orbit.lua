require 'nn'
require 'math'
paths.dofile('task_loss_factory.lua')

local Orbit, parent = torch.class('nn.Orbit', 'nn.Criterion')

function Orbit:__init(l, batch_size, n_classes, is_cuda)
   parent.__init(self)
  
  self.is_avg = false
  self.mul = torch.range(0, (batch_size * n_classes) - 1, n_classes):long()
  self.coeff = 1 / torch.sqrt(2 * math.pi)
  self.is_cuda = is_cuda

  task_loss_factory = TaskLossFactory(l)
  if is_cuda == 'cuda' then
    task_loss_factory:cuda()
  end
end

function Orbit:updateOutput(input, target)    
  local m_scores, y_hat = torch.max(input, 2)
  self.y_hat = y_hat:t()[1]
  if self.is_cuda == 'cuda' then self.y_hat = self.y_hat:cuda()
  else self.y_hat = self.y_hat:float() end

  self.output = task_loss_factory:get_loss(target, self.y_hat)
  if not torch.isTensor(self.output) then
    return self.output
  end
  if self.is_cuda == 'cuda' then self.output = self.output:cuda() end
  return torch.sum(self.output)
end

function Orbit:updateGradInput(input, target)
  -- resize grad_input to a vector instead of matrix
  self.gradInput = torch.zeros(input:size()):float()
  local coeffs_y = torch.zeros(input:size()):float()
  local coeffs_y_hat = torch.zeros(input:size()):float()
  local tmp = torch.ones(target:size()):float()
    
  if self.is_cuda == 'cuda' then
    self.gradInput = self.gradInput:cuda()
    coeffs_y = coeffs_y:cuda()
    coeffs_y_hat = coeffs_y_hat:cuda()
    tmp = tmp:cuda()
  end  
  
  self.gradInput = self.gradInput:view(input:size(1) * input:size(2))

  -- calc the gradients
  local idcs_y = torch.add(target, self.mul):long()
  local idcs_y_hat = torch.add(self.y_hat, self.mul):long()

  coeffs_y = coeffs_y:view(input:size(1) * input:size(2))
  coeffs_y_hat = coeffs_y_hat:view(input:size(1) * input:size(2))

  coeffs_y:indexCopy(1, idcs_y, tmp)
  coeffs_y_hat:indexCopy(1, idcs_y_hat, tmp)

  coeffs_y = coeffs_y:view(input:size(1), input:size(2))
  coeffs_y_hat = coeffs_y_hat:view(input:size(1), input:size(2))

  coeff_2_y = torch.cmul(input, coeffs_y)
  coeff_2_y_hat = torch.cmul(input, coeffs_y_hat)
  coeff_2_y = torch.sum(coeff_2_y, 2)
  coeff_2_y_hat = torch.sum(coeff_2_y_hat, 2)

  res = torch.mul(torch.pow(coeff_2_y - coeff_2_y_hat, 2), 0.5)
  coeff = self.coeff * torch.exp(res):t()[1]

  -- populate them
  self.gradInput:indexCopy(1, idcs_y, torch.cmul(self.output, -1 * coeff))
  self.gradInput:indexCopy(1, idcs_y_hat, torch.cmul(self.output, coeff))

  -- resize grad_input to its original size - matrix
  self.gradInput = self.gradInput:view(input:size(1), input:size(2))
  return self.gradInput
end


