import jax_fem
import os
import inspect

# Find the file
print(os.path.dirname(jax_fem.__file__))

# Search for location_fns usage
src_dir = os.path.dirname(jax_fem.__file__)
with open(os.path.join(src_dir, 'fe.py')) as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'get_boundary_conditions_inds' in line:
        print('\n'.join(lines[i:i+60]))
        break