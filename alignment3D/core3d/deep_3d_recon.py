import torch
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import cv2
from core3d.models import networks  # Use absolute import from core

class Deep3DReconstructor:
    """
    A wrapper for the Deep3DFaceRecon_pytorch model to handle 3D face reconstruction.
    """
    def __init__(self, checkpoint_path: Path, bfm_path: Path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net_recon = self._load_model(checkpoint_path).to(self.device)
        self.facemodel = self._load_bfm(bfm_path)

    def _load_model(self, checkpoint_path: Path):
        """Loads the pre-trained PyTorch model."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
        
        net = networks.ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        net.load_state_dict(state_dict['net_recon'])
        net.eval()
        return net

    def _load_bfm(self, bfm_path: Path):
        """Loads the Basel Face Model (BFM) data robustly."""
        # --- FIX: Prioritize the filename suggested by the user ---
        model_path = bfm_path / 'BFM_front_idx.mat' 
        if not model_path.is_file():
            model_path_alt1 = bfm_path / 'facemodel_info.mat'
            if model_path_alt1.is_file():
                model_path = model_path_alt1
            else:
                model_path_alt2 = bfm_path / 'BFM_front_idx.mat'
                if not model_path_alt2.is_file():
                    raise FileNotFoundError(f"BFM model not found at {bfm_path}. None of the expected files were found.")
                model_path = model_path_alt2
        
        model = loadmat(model_path)

        # Intelligently find keys, ignoring case and variations
        def _find_key(d, s):
            s = s.lower()
            for k in d:
                if s in k.lower():
                    return k
            return None

        shape_mu_key = _find_key(model, 'shapemu')
        shape_pc_key = _find_key(model, 'shapepc')
        exp_pc_key = _find_key(model, 'exppc')
        tri_key = _find_key(model, 'tri')
        keypoints_key = _find_key(model, 'keypoints')

        required_keys_map = {
            'shapeMU': shape_mu_key, 
            'shapePC': shape_pc_key, 
            'expPC': exp_pc_key, 
            'tri': tri_key, 
            'keypoints': keypoints_key
        }
        
        missing_keys = [name for name, key in required_keys_map.items() if key is None]
        if missing_keys:
            raise KeyError(f"Could not find all required BFM keys in {model_path}. Missing: {', '.join(missing_keys)}")

        facemodel = {
            'shapeMU': torch.tensor(model[shape_mu_key], dtype=torch.float32, device=self.device),
            'shapePC': torch.tensor(model[shape_pc_key], dtype=torch.float32, device=self.device),
            'expPC': torch.tensor(model[exp_pc_key], dtype=torch.float32, device=self.device),
            'facelib': {
                'tri': torch.tensor(model[tri_key] - 1, dtype=torch.long, device=self.device),
                'keypoints': torch.tensor(model[keypoints_key] - 1, dtype=torch.long, device=self.device)
            }
        }
        return facemodel

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepares an image for the model (resize, normalize)."""
        im_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        im_tensor = (im_tensor / 255. - 0.5) * 2
        return im_tensor.unsqueeze(0).to(self.device)

    def reconstruct_mesh(self, image_path: str):
        """
        Takes a path to an image, runs reconstruction, and returns a PyVista mesh.
        """
        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")
        
        img = cv2.resize(img, (224, 224))
        im_tensor = self._preprocess_image(img)

        # Run model inference
        with torch.no_grad():
            coeffs = self.net_recon(im_tensor)

        # Reconstruct the 3D shape from the output coefficients
        id_coeffs, exp_coeffs, _, _, _, _ = self._split_coeffs(coeffs)
        face_shape = self._get_shape(id_coeffs, exp_coeffs)
        
        # Get the vertices and faces for the mesh
        vertices = face_shape.squeeze(0).cpu().numpy()
        faces = self.facemodel['facelib']['tri'].cpu().numpy()
        
        # Create and return the PyVista mesh
        import pyvista as pv
        pyvista_faces = np.hstack((np.full((faces.shape[0], 1), 3), faces)).flatten()
        mesh = pv.PolyData(vertices, pyvista_faces)
        return mesh

    def _split_coeffs(self, coeffs):
        """Splits the model's output tensor into its components."""
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80:144]
        tex_coeffs = coeffs[:, 144:224]
        angles = coeffs[:, 224:227]
        gammas = coeffs[:, 227:254]
        translations = coeffs[:, 254:257]
        return id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations

    def _get_shape(self, id_coeffs, exp_coeffs):
        """Reconstructs the 3D face shape from identity and expression coefficients."""
        n_b = id_coeffs.size(0)
        shape_mu = self.facemodel['shapeMU'].expand(n_b, -1, -1)
        exp_pc = self.facemodel['expPC'].expand(n_b, -1, -1)
        
        face_shape = shape_mu + (self.facemodel['shapePC'] @ id_coeffs.unsqueeze(-1)).squeeze(-1).view(n_b, -1, 3)
        face_shape = face_shape + (exp_pc @ exp_coeffs.unsqueeze(-1)).squeeze(-1).view(n_b, -1, 3)
        return face_shape
