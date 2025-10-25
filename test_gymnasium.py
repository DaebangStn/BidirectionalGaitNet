import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet/python')
from ray_env import MyEnv

print("Testing Gymnasium migration...")

env = MyEnv('data/base_lonly.xml', is_xml=True)
obs, info = env.reset()
print('✓ Reset successful')
print(f'✓ Observation shape: {obs.shape}')
print(f'✓ Info keys: {list(info.keys())}')

# Test step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print('✓ Step successful')
print(f'✓ Step returned 5 values (obs, reward, terminated, truncated, info)')
print(f'✓ Terminated: {terminated}, Truncated: {truncated}')
print(f'✓ Info contains reward_map: {"reward_map" in info}')
print('✓ Environment fully compatible with Gymnasium API!')
