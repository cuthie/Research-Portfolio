from cvxopt import matrix,solvers
from numpy import zeros,ones,eye,array,repeat,kron,concatenate as concat
import numpy as np
from numpy.matlib import repmat
import cotc.controller

# MPC using cvxopt to solve 
#
# min     (1/2)x^T Q x + u^T R u + x_N^T Qf + x_N
# subj to Ax = b
#         Gx <= h
#         x_N in Xf
#
# The underlying library solves
#
# min     (1/2)z^T P z + q^T z
# subj to G z^T <= h, z = [x u]
#         Ax = b
#         lb <= x <= ub (?)
#
# The library works with a default CVXOPT solver and with the
# licensed 'mosek' solver.

class MPC(cotc.controller.MPC):
  def __init__(self, A=zeros((1,1)), B=ones((1,1)), Q=ones((1,1)), R=ones((1,1)), Qf=None, Qs=None, horizon=10, state_constraints=None, input_constraints=None, terminal_constraints=None):
    ''' Create an MPC instance
        state_constraints and input_constraints are arrays with two rows. First row are lower constraints and the second are upper.
        Constraint matrices can be set using the specific methods.
    '''
    self._Q = Q
    self._Qf = Qf if not Qf is None else Q
    self._Qs = Qs
    self._R = R
    self.predHorizon = horizon
    self._hx = self._hu = self._xf = None
    self.set_model(A, B)
    self.set_constraints(hx=state_constraints, hu=input_constraints, xf = terminal_constraints) 
    self.xref = zeros((self.num_states,1))

    self.optP = None
    self.optG = None
    self.optA = None

    self.construct()

    self.info = False
    self.max_iterations = 100
    self.solver = None
    self.set_tolerance()

    self.cnt = 0

  def set_tolerance(self, abstol=1e-7, reltol=1e-6, feastol=1e-7):
    if not abstol is None: self.abstol = abstol
    if not reltol is None: self.reltol = reltol
    if not feastol is None: self.feastol = feastol

  @property
  def N(self):
    return self.predHorizon

  def has_soft_constraints(self):
    return self._Qs is not None

  def set_max_iterations(self, n):
    self.max_iterations = n

  def set_costs(self, q = None, r = None, qf = None, qs = None):
    if not q is None: self._Q = q
    if not r is None: self._R = r
    if not qf is None: self._Qf = qf
    if not qs is None: self._Qs = qs
    self.optP = None

  def set_model(self, a,b):
    if not a is None: self.sysA = a
    if not b is None: self.sysB = b
    self.num_states = self.sysB.shape[0]
    self.num_inputs = self.sysB.shape[1]
    self.optA = None
    self.xref = zeros((self.num_states, 1))
    self.set_constraints(hx=self._hx, hu=self._hu, xf=self._xf)

  def set_state_constraints(self, xmin, xmax):
    self._hx = concat( (-array(xmin), array(xmax)), axis=0 )
    self.optG = None

  def set_input_constraints(self, umin, umax):
    self._hu = concat( (-array(umin), array(umax)), axis=0 )
    self.optG = None

  def default_solver(self):
    ''' Set to use the default solver '''
    self.solver = None

  def mosak_solver(self):
    ''' Set to use the mosak solver '''
    self.solver = 'mosak'

  def set_constraints(self, hx=None, hu = None, xf = None):
    n = self.num_states
    m = self.num_inputs
    if hx is None:
      hx = zeros((2*n,))
    if hu is None:
      hu = zeros((2*m,))
    if xf is None:
      xf = hx

    self._hx = array(hx)
    self._hu = array(hu)
    self._xf = array(xf)

    self.set_constraints_on(not hx is None, not hu is None)

    self.optG = None

  def set_constraints_on(self, stateconst, inconst):
    n = self.num_states
    m = self.num_inputs
    if stateconst:
      self._Gx = concat( (-eye(n), eye(n)), axis=0 )
    else:
      self._Gx = zeros((2*n,n))
    if inconst:
      self._Gu = concat( (-eye(m),eye(m)), axis=0 )
    else:
      self._Gu = zeros((2*m,m))
    self.optG = None

  def set_horizon(self, N):
    self.predHorizon = N
    self.optP = self.optA = self.optG = None

  def construct(self, x0=None):
    if x0 is None:
      x0 = zeros((self.num_states,1))

    if self.optP is None: self.construct_cost()
    if self.optG is None: self.construct_inequality()
    if self.optA is None: self.construct_equality()

    self.setx0(x0)

  def construct_cost(self):
    N = self.predHorizon
    n = self.num_states
    m = self.num_inputs

    # Construct the quadratic cost matrix
    optP = concat((
          concat(( kron(eye(N), self._Q), zeros((N*n,m*N)) ), axis=1),
          concat(( zeros((m*N,N*n)), kron(eye(N), self._R)), axis=1)
    ))
    optP[n*(N-1):n*N,n*(N-1):n*N] = self._Qf

    # Construct the linear cost (unused, zero)
    optq = zeros((N*(n+m),1))

    # Soft constraints
    if self._Qs is not None:
      optq = concat( (optq, kron( ones((N,1)), self._Qs.reshape(n,1))) )
      optP = concat( (optP, zeros((N*n, optP.shape[1]))), axis=0 )
      optP = concat( (optP, zeros((optP.shape[0], N*n))), axis=1 )

    self.optq = optq
    self.optP = optP

  def construct_inequality(self):
    N = self.predHorizon
    n = self.num_states
    m = self.num_inputs

    hx = self._hx.copy()
    hu = self._hu.copy()
    hx -= repmat(self.xref,1,2).reshape((2*n,))

    xf = hx.copy()
    # Set and trim terminal set
    if not self._xf is None:
      xf = self._xf.copy()
      xf[0:n] = np.maximum(xf[0:n], hx[0:n])
      xf[n:] = np.minimum(xf[n:], hx[n:])

    hx[0:n] = -hx[0:n]
    hu[0:m] = -hu[0:m]
    xf[0:n] = -xf[0:n]

    # Construct the inequality constraints
    G1 = kron(eye(N), self._Gx)
    G4 = kron(eye(N), self._Gu)
    G2 = zeros((G1.shape[0], G4.shape[1]))
    G3 = zeros((G4.shape[0], G1.shape[1]))
    G12 = concat( (G1, G2), axis=1 )
    G34 = concat( (G3, G4), axis=1 )
    G = concat( (G12, G34), axis=0)

    opth = zeros(((N*(n+m)*2),1))
    opth[0:2*N*n,0] = repmat(hx,1,N)
    opth[2*(N-1)*n:2*N*n,0] = xf
    opth[2*N*n:,0] = repmat(hu, 1, N)

    # Soft constraints
    if self._Qs is not None:
      scm = eye(n)
      for j in range(0,n):
        if self._Qs[j] == 0:  # Zeros soft cost implies hard constraint
          scm[j,j] = 0
      self.scm = scm
      Gscm = kron(eye(N), concat( (-scm, -scm), axis=0)) # The slack additions
      Gscm = concat((Gscm, zeros((m*N*2, n*N)))) # Control actions are hard constraints
      G = concat( (G, Gscm), axis=1 )

      opth = concat( (opth, zeros((N*n,1))) ) # Slack variables
      G = concat( (G, zeros((N*n, G.shape[1]))) )
      G[-N*n:,-N*n:] = -eye(N*n) # Slack variables >= 0

    self.optG = matrix(G)
    self.opth = matrix(opth)
    
  def construct_equality(self):
    N = self.predHorizon
    n = self.num_states
    m = self.num_inputs

    # Construct the equality constraints
    A1 = -eye(N*n) + concat( (concat( (zeros((n, (N-1)*n)), kron(eye(N-1), self.sysA)) ), zeros((N*n, n))), axis=1 )
    A2 = kron(eye(N), self.sysB)
    optA = concat( (A1, A2), axis=1 )

    # Soft constraints
    if self._Qs is not None:
      s = list(optA.shape)
      optA = concat( (optA, zeros((s[0], N*n))), axis=1)

    self.optA = matrix(optA)

  def setx0(self, x0):
    optb = concat((-self.sysA.dot(x0), zeros(((self.predHorizon-1)*self.num_states, 1))))
    self.optb = matrix(optb)

  def calc(self, x, xref):
    n = self.num_states
    m = self.num_inputs
    N = self.predHorizon
  
    # Shift position and constraints
    self.xref = array(xref) 
    self.construct_inequality()
    x0 = array(x)-array(xref)
  
    return self.solve(x0)

  def solve(self, x0=None):
    self.construct()
    if not x0 is None:
      self.setx0(np.array(x0).reshape(self.num_states,1))
    solvers.options['show_progress'] = self.info
    solvers.options['maxiters'] = self.max_iterations
    solvers.options['abstol'] = self.abstol
    solvers.options['reltol'] = self.reltol
    solvers.options['feastol'] = self.feastol
    try:
      self.res = solvers.qp(matrix(self.optP),matrix(self.optq),self.optG,self.opth,self.optA,self.optb,solver=None)
      self.res['N'] = self.predHorizon
      self.res['num_states'] = self.num_states
      self.res['num_inputs'] = self.num_inputs
      return self.res,self.res['status'] == 'optimal'
    except Exception as e:
      raise cotc.controller.InfeasibleException("No feasible path", {})
    self.cnt += 1

  def get_x(self, i):
    return array(MPC.getx(self.res, i)).reshape((self.num_states,))

  def get_u(self, i):
    return array(MPC.getu(self.res, i)).reshape((self.num_inputs,))

  def get_slack(self, i):
    if self._Qs is None: return 0
    idx = self.predHorizon*(self.num_states+self.num_inputs)+i*self.num_states
    sv = array(self.res['x'][idx:idx+self.num_states])
    return self.scm.dot(sv)

  @property
  def _itr(self):
    return self.res['iterations']

  def getx(res, i):
    return res['x'][i*res['num_states']:(i+1)*res['num_states']]

  def getu(res, i):
    return res['x'][res['N']*res['num_states']+i*res['num_inputs']:res['N']*res['num_states']+(i+1)*res['num_inputs']]

  #def predictions(res):
  def predictions(self):
    for i in range(0, res['N']):
    
      yield (array(MPC.getx(res,i)),array(MPC.getu(res,i)))
    
  
  def u_predictions(self):
    aaa=[]
    for i in range(0, self.res['N']):
      aa= array(MPC.getu(self.res, i)).reshape((self.num_inputs,))
      aaa.append(aa)
    return aaa


if __name__ == '__main__':
  import time
  t1 = time.time()
  A = array([[1, 0, 0],[0.05, 1, 0], [-0.008756, -0.3502, 1]]).T
  B = array([[-6.421e-05], [-0.003853], [0.022]])
  mpc = MPC(A=A, B=B, Q=array([[500,0,0],[0,10,0],[0,0,0]]), R=array([[1]]), state_constraints=[-1,-10,-3.14,1,10,3.14], input_constraints=[-10, 10], horizon=30, terminal_constraints=[-0.1,-0.1,-0.1,0.1,0.1,0,1])
  mpc.set_horizon(30)
  t2 = time.time()
  res,success = mpc.solve(x0=[0.53, 0, 0])
  t3 = time.time()
  if not success:
    print('Optimization failed')
  else: 
    for x,u in MPC.predictions(res):
      print(x[0],u)
  print(t3-t1,t3-t2)

