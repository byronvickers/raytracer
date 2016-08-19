import numpy as np
from PIL import Image

def normalise(vec):
  return vec/np.linalg.norm(vec)

def project(vec, onto):
  return onto * np.inner(vec, onto)

class Object():
  def __init__(self, reflectivity=0):
    self.reflectivity = reflectivity
  
  def handle_ray_collision(self, ray):
    if self.reflectivity == 0:
      return self.get_colour(ray.pos)
    inccol = (1-self.reflectivity)*self.get_colour(ray.pos)
    refcol = self.reflectivity*ray.reflect(self.get_normal(ray.pos), self.reflectivity).propagate()
    return refcol + inccol    
    
  def get_colour(self, pos):
    return self.colour

  def get_normal(self, pos):
    raise NotImplementedError()


class Sphere(Object):
  def __init__(self, pos, rad, colour, **kwargs):
    super().__init__(**kwargs)
    self.pos = np.array(pos)
    self.rad = rad
    self.colour = np.array(colour)

  def inside(self, testpos):
    return np.linalg.norm(self.pos - testpos) < self.rad
  
  def raycollide(self, ray):
    if self.inside(ray.pos):
      # ray is inside sphere
      return 0
    
    crel = self.pos - ray.pos
    if np.inner(crel, ray.vec) <= 0:
      # sphere is behind ray
      return np.inf
    
    # shamelessly stolen from http://www.lighthouse3d.com/tutorials/maths/ray-sphere-intersection/
    closest_point = ray.pos + ray.vec*np.inner(crel, ray.vec)
    distance_of_approach = np.linalg.norm(closest_point - self.pos)

    if distance_of_approach >= self.rad:
      # ray will never intersect sphere
      return np.inf

    ray_inside_sphere_dist = np.sqrt(self.rad**2 - distance_of_approach**2)
    return np.linalg.norm(ray.pos - closest_point) - ray_inside_sphere_dist

  def get_normal(self, pos):
    return normalise(pos - self.pos)
    

class HalfSpace(Object):
  def __init__(self, pos, norm, colour, **kwargs):
    super().__init__(**kwargs)
    self.pos = np.array(pos)
    self.norm = normalise(norm)
    self.colour = np.array(colour)

  def inside(self, testpos):
    return np.inner(testpos - self.pos, self.norm) <= 0

  def raycollide(self, ray):
    if self.inside(ray.pos):
      return 0
    
    cos_theta = np.abs( np.inner(self.norm, ray.vec) )
    diff_vec = ray.pos - self.pos
    if cos_theta >= 0.001:
      dist = np.abs(np.inner(diff_vec, self.norm))/cos_theta
      if np.inner(ray.vec, self.norm) <= 0:
        return dist
    return np.inf

  def get_normal(self, pos):
    return self.norm


class Ray():
  def __init__(self, pos, vec, world, ttl=10):
    self.pos = np.array(pos)
    self.vec = normalise(vec)
    self.world = world
    self.ttl = 10
    self.weight = 1

  def propagate(self):
    if self.ttl <= 0 or self.weight <= 0.01:
      return SKYCOLOUR
    world = self.world
    collisions = [obj.raycollide(self) for obj in world]
    index = np.argmin(collisions)
    dist = collisions[index]
    obj = world[index]
    if dist == np.inf:
      return SKYCOLOUR
    else:
      self.pos = self.pos + self.vec * dist
      return obj.handle_ray_collision(self)

  def reflect(self, normal, reflectivity):
    self.vec -= 2*project(self.vec, normal)
    self.ttl -=1
    self.weight *= reflectivity
    # advance to avoid recollision, with thanks to http://www.maths.tcd.ie/~dwmalone/p/rt95.pdf
    self.pos += 0.01*self.vec
    return self

GREY = np.array((128,128,128))
DARK_GREY = np.array((50,50,50))
LIGHT_GREY = np.array((160,160,160))
RED = np.array((250,10,10))
SKYCOLOUR = np.array([50,100,255])
floor = HalfSpace((0,-1,0), (0,1,0), GREY, reflectivity=0.3)
def checkerboard(pos):
  size = 1
  parity = abs(np.floor(pos[0]/size) - np.floor((pos[1]+pos[2])/size)) % 2
  if parity == 1:
    return DARK_GREY
  return LIGHT_GREY
floor.get_colour = lambda pos: checkerboard(pos)

ball1 = Sphere((0,3,10), 2, RED, reflectivity=0.3)
ball2 = Sphere((2,0,5), 1, GREY, reflectivity=0.3)

WORLD = [
  floor,
  ball1,
  ball2,
  ]

positions = [([1,0,9],DARK_GREY),([-3,0.9,5], GREY),([3,1,8], RED),([-1,0,7], DARK_GREY)]
for pos in positions:
  WORLD.append(Sphere(pos[0], 1, pos[1], reflectivity=0.3)) 

h, w = 128, 128
pinhole_depth = 2*h*w/(h+w)
camera_pos = [0,0.8,-2]
img = np.zeros((h, w, 3), dtype=np.uint8)


print('starting...')
for xiter in range(w):
  x = xiter-w/2
  for yiter in range(h):
    y = yiter-h/2
    ray = Ray(camera_pos, [x, y, pinhole_depth], WORLD)
    img[w-yiter-1,xiter] = ray.propagate()

out = Image.fromarray(img)
out.save('my.png')
print('done')
#out.show()
