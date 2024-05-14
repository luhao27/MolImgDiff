from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

#luhao add pdb
import json
pdb_cond = True
import os
# from transformers import BertTokenizer, AutoTokenizer,BertModel, AutoModel 

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None

    # ori code
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    
    # luhao add
    if class_cond:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        class_names = [int(name) for name in set(class_names)]
        min_num = sorted(set(class_names))[0]
        max_num = sorted(set(class_names))[-1]
    
        # print(sorted(set(class_names)))
        # print(max_num, min_num)
        # quit()
        global NUM_CLASSES 
        NUM_CLASSES = max_num + abs(min_num)
        # print(NUM_CLASSES)
        # sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # classes = [sorted_classes[x] for x in class_names]
        classes = [max_num, min_num]

    
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        data_dir = data_dir
    )
    
    
    # ori code
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    # while True:
    #     yield from loader


    # luhao add pdb
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1, data_dir=None):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        
        # ori code
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        
        # luhao add
        self.local_classes = classes
        


        # luhao add pdb
        parent_dir = os.path.dirname(data_dir)
        embedding_pdb = os.path.join(parent_dir, "dict_bedding.json")
        # 读取json文件
        f = open(embedding_pdb)
        self.embedding_pdb = json.load(f)
        f.close()





    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # print("getting.....")
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1
        
        
        # ori code
        # out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        # luhao add
        out_dict = {}
        if self.local_classes is not None:
            # print(path)
            out_dict["y"] = int(path.split("/")[-1].split("_")[0]) + abs(self.local_classes[1])
            # print(out_dict["y"])
            # print(11111111111111111111111111111111)
            # quit()
        
        # print(path)
        # print(path.split("/")[-1].split("_")[-1].split(".")[0])
        pdb_name = path.split("/")[-1].split("_")[-1].split(".")[0]

        pdb_bebing = self.embedding_pdb[pdb_name]
        # print(len(pdb_bebing))
        # print(len(pdb_bebing[0]))
        # print(len(pdb_bebing[0][0]))
        # print(len(pdb_bebing[0][0][0]))
        # print(len(pdb_bebing[0][0][0][0]))

        # luhao add pdb
        # if self.pdb_name is not None:
        #     print(self.pdb_name[idx].strip())
        #     fasta = self.fasta[self.pdb_name[idx].strip()]
        #     print(fasta)
        #     embedding = self.get_emb(fasta)
        #     out_dict["embedding"] = embedding

        return np.transpose(arr, [2, 0, 1]), out_dict, np.array(pdb_bebing).astype(np.float32)
