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

E, nu = 1000, 0.3
#rho = 2.7*(100**3)/1000
rho = 1
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


Time = 8       #Final Time
num_steps = 800  #Number of time Steps
dtt = Time/num_steps  #Time Step Size

T = Constant((0, 0, -0.05))

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

vtkfile = File('Elastodynamics/Solution.pvd')

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




