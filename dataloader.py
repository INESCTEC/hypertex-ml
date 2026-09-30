import numpy as np
import h5py
import random
import torch

class HDF5HSIDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_file, window_size, step):
        super(HDF5HSIDataset, self).__init__()

        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if isinstance(step, int):
            step = (step, step)

        # Data Config
        self.window_size = window_size
        self.step = step

        # List of files
        self.hdf5_data = h5py.File(hsi_file, 'r')
        self.grouped_indices = self._generate_indices()

        self.indices = [ idx for group in self.grouped_indices for idx in group ]

    def _generate_indices(self):
        h, w = self.window_size
        step_h, step_w = self.step
        
        indices = []
        tmp_indices = []
        for key, value in self.hdf5_data["gt"].items():
            label_mask = np.any((np.array(value) > 0), axis=2)

            H, W = value.shape[:2]
            offset_w = (W - w) % step_w
            offset_h = (H - h) % step_h

            for y in range(0, H - h + offset_h + 1, step_h):
                if (y + (h//2) > H) or (y - (h//2) <= 0):
                    # y = H - h
                    continue
                for x in range(0, W - w + offset_w + 1, step_w):
                    if (x + (w//2) > W) or (x - (w//2) <= 0):
                        # x = W - w
                        continue
                    if label_mask[y,x]:
                        tmp_indices.append((key, y, x))

            indices.append(tmp_indices)
            tmp_indices = []

        return indices 

    
    def create_split(
        self,
        val_split=0.2,
        samples_per_image=None,
    ):
        train_indices = []
        val_indices = []

        if samples_per_image is None:

            indices = list(range(len(self)))

            random.shuffle(indices)

            n_val = int(len(indices) * val_split)

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            return train_indices, val_indices

        start_idx = 0

        for image_indices in self.grouped_indices:

            image_positions = list(
                range(
                    start_idx,
                    start_idx + len(image_indices)
                )
            )

            start_idx += len(image_indices)

            random.shuffle(image_positions)

            n_samples = min(samples_per_image, len(image_positions))

            n_val = int(n_samples * val_split)
            n_train = n_samples - n_val

            val_subset = image_positions[:n_val]
            train_subset = image_positions[n_val:n_val + n_train]

            train_indices.extend(train_subset)
            val_indices.extend(val_subset)

        return train_indices, val_indices
                          
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        key, y, x = self.indices[idx]

        h, w = self.window_size
        
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x1 + w, y1 + h
        
        data = self.hdf5_data["data/"+key][y1:y2,x1:x2].transpose((2, 0, 1))
        target =self.hdf5_data["gt/"+key][y, x]

        data = np.asarray(np.copy(data), dtype="float32")
        target = np.asarray(np.copy(target), dtype="float32")

        data = torch.from_numpy(data)
        target = torch.from_numpy(target)

        if h == 1 and w == 1:
            # 2D Spectral Signal
            data = data.squeeze(-1).squeeze(-1)
            data = data.unsqueeze(0)
        else:
            # 3D Spectral Signal
            data = data.unsqueeze(0) # (1, depth, H, W)
        
        return data, target
    

