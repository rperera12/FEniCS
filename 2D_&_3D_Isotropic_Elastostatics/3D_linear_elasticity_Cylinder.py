from __future__ import print_function
from fenics import *
from ufl import nabla_div
from dolfin import *
from ufl import nabla_div
import dolfin as dolfin
from mshr import *


L, B, H = 20., 0.5, 1.
Nx = 200
Ny = int(B/L*Nx)+1
Nz = int(H/L*Nx)+1

E, nu = 1e5, 0.
rho = 2*5
f = Constant((0, 0, -rho*2.5))
T = Constant((0, 0, 0))
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

cylinder = Cylinder(Point(0.,0.,0.),Point(L,0,0), 2,2)
geometry = cylinder
mesh = generate_mesh(geometry,40)
File("cylinder.pvd") << mesh
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)

tol = 1E-14
def left(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0.,0.,0.)), left)

def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(v):
    
    return lmbda*nabla_div(u)*Identity(d) + 2*mu*eps(u)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
a = inner(sigma(u), eps(v))*dx
l = dot(f,v)*dx + dot(T,v)*ds

u = Function(V, name="Displacement")
solve(a == l, u, bc)

u_magnitude = sqrt(dot(u,u))
V = FunctionSpace(mesh, 'P', 1)
u_magnitude = Function(V, name="Magnitude of Displacement")
u_magnitude = project(u_magnitude, V)

File('3D_Cylinder_Elasicity/Displacement.pvd') << u
File('3D_Cylinder_Elasicity/Magnitude of Displacement.pvd') << u_magnitude
