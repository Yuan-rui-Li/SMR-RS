import torch

def random_choice(num, gallery=None, num_gallery:int=None):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.
            num_gallery:(int):number of elements in gallery

        Returns:
            Tensor or ndarray: sampled indices.
        """
        
        if num_gallery is None:
            assert len(gallery) >= num
            is_tensor = isinstance(gallery, torch.Tensor)
            if not is_tensor:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                else:
                    device = 'cpu'
                gallery = torch.tensor(gallery, dtype=torch.long, device=device)
            # This is a temporary fix. We can revert the following code
            # when PyTorch fixes the abnormal return of torch.randperm.
            # See: https://github.com/open-mmlab/mmdetection/pull/5014
            """
            torch.randperm(n):将0~n-1(包括0和n-1)随机打乱后获得的数字序列,函数名是random permutation缩写

            【sample】

            torch.randperm(10)
            ===> tensor([2, 3, 6, 7, 8, 9, 1, 5, 0, 4])
            """
            perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
            rand_inds = gallery[perm]
            if not is_tensor:
                rand_inds = rand_inds.cpu().numpy()
            return rand_inds
        else:
            perm_index = torch.randperm(num_gallery)[:num]
            perm_index = perm_index.cpu().numpy()
        return perm_index
