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

E, nu = 68e9, 0.3
rho = 2.7*(100**3)/1000
f = Constant((0, 0, 0))
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

mesh = BoxMesh(Point(0.,0.,0.),Point(L,B,H), Nx, Ny, Nz)
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)

tol = 1E-14
# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return near(x[0], 1.) and on_boundary

bc = DirichletBC(V, Constant((0.,0.,0.)), left)

def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(v):
    return lmbda*nabla_div(u)*Identity(d) + 2*mu*eps(u)


beta = 0.001
R0 = 10
T=Expression(("0","100*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))","0"),degree=1, beta=beta, R0=R0)


##################################################################################


u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
a = inner(sigma(u), eps(v))*dx
l = dot(f,v)*dx + dot(T,v)*ds

u = Function(V, name="Displacement")
solve(a == l, u, bc)

stress = sigma(u)
Vs = TensorFunctionSpace(mesh, 'P', 1)
stress2 = Function(Vs, name="Stress")
stress2 = project(stress, Vs)
File('Elastodynamics/Stress.pvd') << stress2

T = 10          #Final Time
num_steps = 2000   #Number of time Steps
dtt = T/num_steps  #Time Step Size

u_n = u
u_nminus = u

u = TrialFunction(V)
F = rho*dot(v,u)*dx + (inner(sigma(u),eps(v))*dtt**2)*dx-rho*dot((2*u_n-u_nminus),v)*dx
#L = rho*dot((2*u_n-u_nminus),v)*dx
a, L = lhs(F), rhs(F)

u = Function(V)

vtkfile = File('Elastodynamics/Solution.pvd')


# Initial Conditions are already defined now lets apply propagation


#F = rho*dot(v,u)*dx + (inner(sigma(u),eps(v))*dtt**2)*dx-rho*dot((2*u_n-u_nminus),v)*dx
t=0
for n in range(num_steps):

		
	# Update current time
	t += dtt
	
	# Compute solution
	solve(a == L, u,bc)
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




