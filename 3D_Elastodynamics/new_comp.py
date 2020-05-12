from __future__ import print_function
from fenics import *
from ufl import nabla_div
from dolfin import *
from ufl import nabla_div
import dolfin as dolfin
from mshr import *
import time

L, B, H = 20., 0.1, 1.
Nx = 200
Ny = int(B/L*Nx)+1
Nz = int(H/L*Nx)+1

print(Nx)
print(Ny)
print(Nz)

tol = 1E-14
E_0 = 0.01e7
E_1 = 200e9


E = Expression('x[1]>=0.04-tol && x[1]<=0.06+tol ? E_1 : E_0', degree=0, tol=tol, E_1=E_1, E_0=E_0)

nu_0 = 0.3
nu_1 = 0.3
nu = Expression('x[1]>=0.04-tol && x[1]<=0.06+tol ? nu_1 : nu_0', degree=0, tol=tol,  nu_1=nu_1, nu_0=nu_0)

rho_0 = 3.85
rho_1 = 7.75
rho = Expression('x[1]>=0.04-tol && x[1]<=0.06+tol ? rho_1 : rho_0', degree=0, tol=tol,  rho_1=rho_1, rho_0=rho_0)

f = Constant((0, 0, 0))
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

#mesh = BoxMesh(Point(0.,0.,0.),Point(L,B,H), Nx, Ny, Nz)
mesh = BoxMesh(Point(0.,0.,0.),Point(1.,0.1,0.04), 18 , 7, 4)
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)

tol = 1E-14
# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return on_boundary and near(x[0], 1.) #and near(x[1], 0.1) 

bc = DirichletBC(V, Constant((0.,0.,0.)), left)

def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(u):
    return lmbda*nabla_div(u)*Identity(d) + 2*mu*eps(u)


Time = 0.5       #Final Time
num_steps = 20000  #Number of time Steps
dtt = Time/num_steps  #Time Step Size

T = Constant((0, 0, -10000))

boundary_subdomains = MeshFunction("size_t",mesh,mesh.topology().dim()-1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains,3)


dss = ds(subdomain_data=boundary_subdomains)

##################################################################################


u_init = TrialFunction(V)
d = u_init.geometric_dimension()
v = TestFunction(V)
a = inner(sigma(u_init), eps(v))*dx
l = dot(f,v)*dx + dot(T,v)*dss  #dss(3)

u_init = Function(V, name="Displacement")
solve(a == l, u_init, bc)


##########################################################################################
##########################################################################################




def eps1(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma1(u):
    return lmbda*nabla_div(u)*Identity(d) + 2.0*mu*eps1(u)

u_n = Function(V)
u_nminus = Function(V)

u_n.assign(u_init)
u_nminus.assign(u_init)

u = TrialFunction(V)
F = rho*dot(v,u)*dx + (inner(sigma1(u),eps1(v))*(dtt**2.0))*dx-rho*dot((2.0*u_n-u_nminus),v)*dx

#L = rho*dot((2*u_n-u_nminus),v)*dx
a1, L = lhs(F), rhs(F)
u = Function(V)

vtkfile = File('New_Comp/Solution.pvd')

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



#u_magnitude = sqrt(dot(u,u))
#V = FunctionSpace(mesh, 'P', 1)
#u_magnitude = Function(V, name="Magnitude of Displacement")
#u_magnitude = project(u_magnitude, V)

#File('3D_Beam_Elasicity/Displacement.pvd') << u
#File('3D_Beam_Elasicity/Magnitude of Displacement.pvd') << u_magnitude




