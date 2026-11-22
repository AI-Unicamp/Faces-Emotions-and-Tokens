def _extract_flame_params(self, npz_path, chunk_idx):
        """Extract FLAME parameters for chunk"""
        data = np.load(npz_path, allow_pickle=True)
        expressions = data['expressions']  # (num_frames, 100)
        poses = data['poses']  # (num_frames, 165)
        
        # Calculate frame range
        step = self.chunk_duration - self.overlap_duration
        start_time = chunk_idx * step
        end_time = start_time + self.chunk_duration
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        end_frame = min(end_frame, expressions.shape[0], poses.shape[0])
        
        # Extract FLAME params
        expressions_chunk = expressions[start_frame:end_frame]  # (T, 100)
        jaw_chunk = poses[start_frame:end_frame, 66:69]  # (T, 3)
        
        # Concatenate to FLAME (103)
        flame_params = np.concatenate([expressions_chunk, jaw_chunk], axis=1)
        
        return flame_params.astype(np.float32)


def flame_to_arkit_blendshapes(flame_params, transform_matrix):
    """
    Transform FLAME parameters to ARKit blendshapes
    
    Args:
        flame_params: (*, 103) tensor/array of FLAME parameters
        transform_matrix: (103, 51) transformation matrix
    
    Returns:
        arkit_blendshapes: (*, 51) tensor/array
    """
    if isinstance(flame_params, torch.Tensor):
        if not isinstance(transform_matrix, torch.Tensor):
            transform_matrix = torch.from_numpy(transform_matrix).float().to(flame_params.device)
        return torch.matmul(flame_params, transform_matrix)
    else:
        if isinstance(transform_matrix, torch.Tensor):
            transform_matrix = transform_matrix.cpu().numpy()
        return np.dot(flame_params, transform_matrix)

def load_flame_to_arkit_transform(path='arkit_to_flame.npy'):
    """
    Load and compute FLAME to ARKit transformation matrix
    
    Args:
        path: Path to arkit_to_flame.npy (shape: 51, 103)
    
    Returns:
        transform: (103, 51) transformation matrix for FLAME -> ARKit
    """
    arkit_to_flame = np.load(path)  # (51, 103)
    flame_to_arkit = np.linalg.pinv(arkit_to_flame)  # (103, 51)
    return flame_to_arkit

# Extract FLAME
flame_params = _extract_flame_params(
    sample['npz_path'],
    sample['chunk_idx']
)

flame_to_arkit = load_flame_to_arkit_transform(arkit_transform_path)

# Transform to ARKit
arkit_blendshapes = flame_to_arkit_blendshapes(
    flame_params,
    flame_to_arkit
)