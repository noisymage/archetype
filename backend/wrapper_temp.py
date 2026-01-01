class SMPLestXWrapper:
    """
    Wrapper for SMPLest-X inference to handle complex config and model setup.
    """
    def __init__(self, device: str):
        self.device = device
        self.cfg = None
        self.model = None
        self.smpl_x = None
        self.initialized = False
        
    def initialize(self) -> bool:
        if self.initialized:
            return True
            
        if not SMPLEST_X_AVAILABLE:
            return False
            
        try:
            # Paths
            ckpt_path = MODELS_DIR / "smplx/pretrained/smplest_x_h.pth.tar"
            human_model_path = MODELS_DIR / "smplx/body_models"
            
            if not ckpt_path.exists():
                logger.error(f"SMPLest-X checkpoint missing: {ckpt_path}")
                return False
                
            # Scaffold Config (mimic config_smplest_x_h.py)
            conf_dict = {
                "model": {
                    'model_type': 'vit_huge',
                    "pretrained_model_path": str(ckpt_path),
                    "human_model_path": str(human_model_path),
                    'encoder_config': {
                        'num_classes': 80, 'task_tokens_num': 80, 'img_size': (256, 192),
                        'patch_size': 16, 'embed_dim': 1280, 'depth': 32, 'num_heads': 16,
                        'ratio': 1, 'use_checkpoint': False, 'mlp_ratio': 4, 'qkv_bias': True, 'drop_path_rate': 0.55
                    },
                    'decoder_config': {'feat_dim': 1280, "dim_out": 512, 'task_tokens_num': 80},
                    'input_img_shape': (512, 384),
                    'input_body_shape': (256, 192),
                    'output_hm_shape': (16, 16, 12),
                    'focal': (5000, 5000), 'princpt': (192 / 2, 256 / 2),
                    'camera_3d_size': 2.5,
                },
                "test": {"test_batch_size": 1}
            }
            
            self.cfg = SMPLConfig(conf_dict)
            
            # Init Human Model
            self.smpl_x = SMPLX(str(human_model_path))
            
            # Init Network
            logger.info("Loading SMPLest-X model (this may take a moment)...")
            self.model = get_model(self.cfg, 'test')
            
            # Load Checkpoint
            if self.device == 'mps':
                 # MPS sometimes needs CPU load -> move
                 ckpt = torch.load(str(ckpt_path), map_location='cpu')
            else:
                 ckpt = torch.load(str(ckpt_path), map_location='cpu')
                 
            # State dict processing (from Tester._make_model)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['network'].items():
                if 'module' not in k: k = 'module.' + k
                k = k.replace('backbone', 'encoder').replace('body_rotation_net', 'body_regressor').replace(
                    'hand_rotation_net', 'hand_regressor')
                new_state_dict[k] = v
                
            # Load with strict=False as per their code
            self.model.load_state_dict(new_state_dict, strict=False)
            
            # Move to device
            if self.device == 'cuda':
                self.model = self.model.cuda()
            elif self.device == 'mps':
                self.model = self.model.to('mps')
            else:
                self.model = self.model.cpu()
                
            self.model.eval()
            self.initialized = True
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize SMPLest-X: {e}")
            return False

    def run_inference(self, image: np.ndarray, bbox: list) -> dict:
        """
        Run inference on a single image with a given bounding box.
        bbox: [x1, y1, x2, y2]
        """
        if not self.initialized:
            return None
            
        try:
            import torch
            from torchvision import transforms
            
            # Prepare bounding box (xyxy -> xywh)
            # Their code expects: box_xywh[2] = width, box_xywh[3] = height
            bbox_xywh = np.array([bbox[0], bbox[1], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1])])
            
            # Preprocess using their utils
            original_img_height, original_img_width = image.shape[:2]
            proc_bbox = process_bbox(bbox_xywh, original_img_width, original_img_height, 
                                     input_img_shape=self.cfg.model.input_img_shape)
            
            img_patch, _, _ = generate_patch_image(cvimg=image.copy(), bbox=proc_bbox, 
                                                   scale=1.0, rot=0.0, do_flip=False, 
                                                   out_shape=self.cfg.model.input_img_shape)
            
            # Transform
            transform = transforms.ToTensor()
            img_tensor = transform(img_patch.astype(np.float32))/255
            img_tensor = img_tensor.unsqueeze(0) # Batch dim
            
            if self.device == 'cuda':
                img_tensor = img_tensor.cuda()
            elif self.device == 'mps':
                 img_tensor = img_tensor.to('mps')
            
            # Run Model
            with torch.no_grad():
                inputs = {'img': img_tensor}
                out = self.model(inputs, {}, {}, 'test')
            
            # Extract Results
            betas = out['smplx_shape'].reshape(-1, 10).cpu().numpy()[0]
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
            
            return {
                'betas': betas.tolist(),
                'mesh': mesh, 
                'volume_proxy': self._calc_volume(mesh)
            }
            
        except Exception as e:
            logger.error(f"Inference run failed: {e}")
            return None

    def _calc_volume(self, mesh_vertices):
        """Estimate volume from mesh vertices (proxy)."""
        # Simple bounding box volume of the mesh for now
        try:
            min_coords = np.min(mesh_vertices, axis=0)
            max_coords = np.max(mesh_vertices, axis=0)
            dims = max_coords - min_coords # [width, height, depth]
            # Volume of bounding box
            vol = dims[0] * dims[1] * dims[2]
            return float(vol)
        except:
            return 0.0
