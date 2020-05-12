from __future__ import print_function
from fenics import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import dolfin as d


# variables
mu = 1
rho = 1
beta = 1.25
lambda_ = beta
g = 9.81

# create mesh
comm = d.MPI.comm_world
mesh = UnitDiscMesh.create(comm, 20, 2, 3) 
# create function space
V = VectorFunctionSpace(mesh, 'Lagrange', 3) # defining a vector valued function space over the mesh with a lagrangian finite elements of degree 1


# boundary condition
tol = 1E-14

def clamped_boundary(x, on_boundary):
     return on_boundary and x[0] < tol # this should "clamp" the part of the disc left of the x axis

bc = DirichletBC(V, (0, 0, 0), clamped_boundary) # create dirichlet boundary condition where u = (0,0,0) on the clamped boundary


# define stress and strain
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T) # define the symmetric part of the gradient of a vector function u(a tensor)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u) # defining the function for sigma dependent on the displacement u


# define trial and test functions
u = TrialFunction(V)
d = u.geometric_dimension() # not sure what this is doing
v = TestFunction(V) #defined both the test and trial function over the function space V

# variational problem
f = Constant((0, 0, -rho*g)) # force per unit body mass
T = Constant((0, 0, 0)) # sigma dot n
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds # variational equation

# compute solution
u = Function(V)
solve(a == L, u, bc, solver_parameters={'linear_solver':'mumps'})

# save solution for paraview
vtkfile = File('Unit_Disk_Elasticity.pvd')
vtkfile << u
