import numpy as np
class NPValue:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad=np.zeros_like(data)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label=label


  def __repr__(self):
    return f"NPValue(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other,NPValue) else NPValue(other)
    out = NPValue(np.add(self.data,other.data), (self, other), '+')
    def _backward():
      self.grad += 1.0*out.grad
      other.grad += 1.0*out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other,NPValue) else NPValue(other)
    out = NPValue(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data*out.grad
      other.grad += self.data*out.grad
    out._backward = _backward
    return out
      
 def __matmul__(self, other):
    other = other if isinstance(other,NPValue) else NPValue(other)
    out = NPValue(self.data @ other.data, (self, other), '@')
    def _backward():
      self.grad += out.grad@((other.data).T)
      other.grad += ((self.data).T)@out.grad
    out._backward = _backward
    return out

  def pow(self,other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = NPValue(self.data**other, (self,), f'**{other}')
    def _backward():
      self.grad += other*(self.data**(other-1))*out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    other = other if isinstance(other, NPValue) else NPValue(other) # Ensure other is a Value object
    return self * other.pow(-1.0) # Correctly implement division using multiplication and power

  def __sub__(self,other):
    other = other if isinstance(other,NPValue) else NPValue(other)
    out = NPValue(self.data - other.data, (self, other), '-')
    # Backward pass for subtraction is similar to addition
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad -= 1.0 * out.grad # Negative gradient for the subtrahend
    out._backward = _backward
    return out


  def tanh(self):
    n = self.data
    t=NPValue((np.exp(2*n)-1)/(np.exp(2*n)+1))
    out=NPValue(t.data,(self,),'tanh')
    def _backward():
      self.grad += (1-t.data**2)*out.grad
    out._backward = _backward
    return out

  def exp(self):
    x=self.data
    out=NPValue(np.exp(x),(self,),'exp')
    def _backward():
      self.grad += out.data*out.grad
    out._backward = _backward
    return out

  def backward(self):
    topo=[]
    visited=set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad=np.ones_like(self.data)
    for node in reversed(topo):
      node._backward()
