import numpy as np

a = np.load('./dataset.npz')

betas = a['betas']
poses = a['poses']
frontal_sample_points = a['frontal_sample_points']
lateral_sample_points = a['lateral_sample_points']

print(betas.shape)
print(poses.shape)
print(frontal_sample_points.shape)
print(lateral_sample_points.shape)

np.savez('./dataset_modified_sample_point_no_vertices.npz', betas=betas, poses=poses, frontal_sample_points=frontal_sample_points, lateral_sample_points=lateral_sample_points)

b = np.load('./dataset_modified_sample_point_no_vertices.npz')
print(b.files)
print(b['betas'].shape)