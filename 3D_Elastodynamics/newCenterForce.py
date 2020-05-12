from __future__ import print_function
from fenics import *
from ufl import nabla_div
from dolfin import *
from ufl import nabla_div
import dolfin as dolfin
from mshr import *


L, B, H = 10, 10, 2
Nx = 200
Ny = int(B/L*Nx)+1
Nz = int(H/L*Nx)+1

E, nu = 0.01E-7, 0.2
rho = 7.75
f = Constant((0, 0, 0))
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

tol = 1e-14
T = Constant((0,0,0.5))

mesh = BoxMesh(Point(0.,0.,0.),Point(L,B,H), Nx, Ny, Nz)
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)

tol = 1E-14
def boundary(x, on_boundary):
     return on_boundary and near(x[0], 0) or near(x[0], L) or near(x[1], 0) or near(x[1], B) 

def fboundary(x, on_boundary):
    return on_boundary and x[0]<6 and x[0]>4 and x[1]<6 and x[1]>4


boundary_subdomains = MeshFunction("size_t",mesh,mesh.topology().dim()-1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(fboundary)
force_boundary.mark(boundary_subdomains,3)
dss = ds(subdomain_data=boundary_subdomains)

bc = DirichletBC(V, Constant((0.,0.,0.)), boundary)

def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(u):
    return lmbda*nabla_div(u)*Identity(d) + 2*mu*eps(u)

u_init = TrialFunction(V)
d = u_init.geometric_dimension()
v = TestFunction(V)
a = inner(sigma(u_init), eps(v))*dx
l = dot(f,v)*dx + dot(T,v)*dss(3)

u_init = Function(V, name="Displacement")
solve(a == l, u_init, bc)


File('CenterForce/Displacement.pvd') << u_init



Time = 0.5       #Final Time
num_steps = 80000  #Number of time Steps
dtt = Time/num_steps  #Time Step Size



##########################################################################################
##########################################################################################

u_n = Function(V)
u_nminus = Function(V)

u_n.assign(u_init)
u_nminus.assign(u_init)

u = TrialFunction(V)


def eps1(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma1(u):
    return lmbda*nabla_div(u)*Identity(d) + 2.0*mu*eps1(u)


F = rho*dot(v,u)*dx + (inner(sigma1(u),eps1(v))*(dtt**2.0))*dx-rho*dot((2.0*u_n-u_nminus),v)*dx

#L = rho*dot((2*u_n-u_nminus),v)*dx
a1, L = lhs(F), rhs(F)
u = Function(V)

vtkfile = File('CenterForce/Solution.pvd')

# Initial Conditions are already defined now lets apply propagation


#F = rho*dot(v,u)*dx + (inner(sigma(u),eps(v))*dtt**2)*dx-rho*dot((2*u_n-u_nminus),v)*dx
t=0.0
for n in range(num_steps):

		
	# Update current time
	t += dtt
	
	# Compute solution
	solve(a1 == L, u,bc)
	vtkfile << (u,t)
	# Update previous solution
	#u_nplus.assign(u)

	u_nminus.assign(u_n)
	u_n.assign(u)


