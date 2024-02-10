#Install from GitHub
pip install git+https://github.com/dmarcondes/JINNAX
from jinnax import data as jd

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

dat = jd.generate_dDimdataPINN(u,0,1,0,1,100,100,100,d = 2,posx = 'grid',post = 'grid',posc = 'grid',sigmaB = 0,sigmaI = 10,sigmaS = 0)
dat['collocation']
dat['uboundary']
for name,value in dat.items():
    print(name)
    print(value.shape)
with jnp.printoptions(threshold=jnp.inf):
    print(x_boundary)

files_path = ['/home/dmarcondes/Dropbox/Diego/Profissional/Code/Experiments/DMNN_experiments/data/material_science/Phi_comp_' + str(i+1) + '.jpg' for i in range(10)]
f = ['/home/dmarcondes/Pictures/d&l.jpeg']
f = jd.png_to_jnp(f)
f = f[0,:,:,0]

x = jax.random.unifo

rm(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = 0,maxval = 1,shape = (10,5))
f = fconNN([2,40,40,40,1])
params = f['params']
def k(x):
    return f['forward'](params,x)
forward(params,x)
