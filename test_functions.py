def u(x,t):
    return x[0] + x[1] + x[3] + t**2

xlo = 0
xhi = 1
tlo = 0
thi = 1
Nx = 10
Nt = 10
Nc = 10
sigmaB = 10
sigmaI = 0
sigmaS = 0

dat = generate_dDimdataPINN(u,0,1,0,1,100,100,100,d = 2,posx = 'grid',post = 'grid',posc = 'grid',sigmaB = 0,sigmaI = 10,sigmaS = 0)
dat['collocation']
dat['uboundary']
for name,value in dat.items():
    print(name)
    print(value.shape)
with jnp.printoptions(threshold=jnp.inf):
    print(x_boundary)
