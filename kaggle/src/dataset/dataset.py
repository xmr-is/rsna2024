import pandas as pd
import numpy as np
import glob
import torch
from typing import Tuple, List
from PIL import Image
import pydicom
import cv2

from torch.utils.data import Dataset
from run.config import TrainConfig, InferenceConfig
from src.utils.environment_helper import EnvironmentHelper, InferenceEnvironmentHelper

class TrainDataset(Dataset):
    def __init__(
            self, 
            cfg: TrainConfig,
            df: pd.DataFrame, 
            phase='train', 
            transform=None
        ) -> None:
        self.cfg = cfg
        self.df = df
        self.phase = phase
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        512x512x30の3次元の配列を作成、
        in_chansはaxial,sagittal_t1,t2の3つそれぞれの画像10枚を組み合わせるための入れ物
        (512,512,0-9) -> Sagittal T1
        (512,512,10-19) -> Sagittal T2/STIR
        (512,512,20-29) -> Axial T2
        """
        x = np.zeros((
            512,
            512,
            self.cfg.model.params.in_channels), 
            dtype=np.uint8
        )

        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)
        
        # Sagittal T1        
        for i in range(0, 10, 1):
            try:
                p = f'{self.cfg.directory.image_dir}/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                pass
                # raise RuntimeError(f'failed to load on {st_id}, Sagittal T1')
        
        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'{self.cfg.directory.image_dir}/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                pass
                # raise RuntimeError(f'failed to load on {st_id}, Sagittal T2/STIR')
        
        # Axial T2
        axt2 = glob.glob(f'{self.cfg.directory.image_dir}/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)
        
        # 10枚分抽出するため全枚数を10等分
        step = len(axt2) / 10.0
        # 真ん中を求めて、10等分のうち1個目のindexを取得
        st = len(axt2)/2.0 - 4.0*step 
        # 10等分のうち10個目の区切りのindexを取得(+0.0001は最後のindexを配列に含めるため)
        end = len(axt2)+0.0001
                
        for i, j in enumerate(np.arange(st, end, step)):
            try:
                # 60枚あるAxial_T2から10枚を６スライスごとに抽出する、
                # メモリ消費を抑える、計算コストを抑えるため？
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+20] = img.astype(np.uint8)
            except:
                pass
                # raise RuntimeError(f'failed to load on {st_id}, Sagittal T2/STIR')

        assert np.sum(x)>0
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, label

class ValidDataset(Dataset):
    def __init__(
            self, 
            cfg: TrainConfig,
            df: pd.DataFrame, 
            phase='valid', 
            transform=None
        ) -> None:
        self.cfg = cfg
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        512x512x30の3次元の配列を作成、
        in_chansはaxial,sagittal_t1,t2の3つそれぞれの画像10枚を組み合わせるための入れ物
        (512,512,0-9) -> Sagittal T1
        (512,512,10-19) -> Sagittal T2/STIR
        (512,512,20-29) -> Axial T2
        """
        x = np.zeros((
            512,
            512,
            self.cfg.model.params.in_channels), 
            dtype=np.uint8
        )

        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)
        
        # Sagittal T1        
        for i in range(0, 10, 1):
            try:
                p = f'{self.cfg.directory.image_dir}/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                pass
                # raise RuntimeError(f'failed to load on {st_id}, Sagittal T1')
                
        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'{self.cfg.directory.image_dir}/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                pass
                # raise RuntimeError(f'failed to load on {st_id}, Sagittal T2/STIR')
                
        # Axial T2
        axt2 = glob.glob(f'{self.cfg.directory.image_dir}/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)
        
        # 10枚分抽出するため全枚数を10等分
        step = len(axt2) / 10.0
        # 真ん中を求めて、10等分のうち1個目のindexを取得
        st = len(axt2)/2.0 - 4.0*step 
        # 10等分のうち10個目の区切りのindexを取得(+0.0001は最後のindexを配列に含めるため)
        end = len(axt2)+0.0001
                
        for i, j in enumerate(np.arange(st, end, step)):
            try:
                # 60枚あるAxial_T2から10枚を６スライスごとに抽出する、
                # メモリ消費を抑える、計算コストを抑えるため？
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+20] = img.astype(np.uint8)
            except:
                pass
                # raise RuntimeError(f'failed to load on {st_id}, Sagittal T2/STIR')
            
        assert np.sum(x)>0
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, label
    
class InferenceDataset(Dataset):
    def __init__(
            self, 
            cfg: InferenceConfig, 
            df: pd.DataFrame, 
            study_ids: List[int], 
            phase='inference', 
            transform=None
        ):
        self.cfg = cfg
        self.df = df
        self.study_ids = study_ids
        self.transform = transform
        self.phase = phase
        self.env = InferenceEnvironmentHelper(self.cfg)
    
    def __len__(self):
        return len(self.study_ids)
    
    def get_img_paths(
            self, 
            study_id, 
            series_desc
        ) -> Tuple[torch.Tensor, List[str]]:
        pdf = self.df[self.df['study_id']==study_id]
        pdf_ = pdf[pdf['series_description']==series_desc]
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(f'{self.cfg.directory.base_dir}/test_images/{study_id}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=self.env.natural_keys)
            allimgs.extend(pimgs)
            
        return allimgs
    
    def read_dcm_ret_arr(self, src_path) -> np.ndarray:
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        img = cv2.resize(
            image, 
            (self.cfg.dataset.image_size, 
             self.cfg.dataset.image_size),
            interpolation=cv2.INTER_CUBIC
        )
        assert img.shape==(self.cfg.dataset.image_size, self.cfg.dataset.image_size)
        return img
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        x = np.zeros((
            self.cfg.dataset.image_size, 
            self.cfg.dataset.image_size, 
            self.cfg.model.params.in_channels
            ), dtype=np.uint8
        )
        st_id = self.study_ids[idx]        
        
        # Sagittal T1
        allimgs_st1 = self.get_img_paths(st_id, 'Sagittal T1')
        if len(allimgs_st1)==0:
            print(st_id, ': Sagittal T1, has no images')
        
        else:
            step = len(allimgs_st1) / 10.0
            st = len(allimgs_st1)/2.0 - 4.0*step
            end = len(allimgs_st1)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st1[ind2])
                    x[..., j] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass
            
        # Sagittal T2/STIR
        allimgs_st2 = self.get_img_paths(st_id, 'Sagittal T2/STIR')
        if len(allimgs_st2)==0:
            print(st_id, ': Sagittal T2/STIR, has no images')
        else:
            step = len(allimgs_st2) / 10.0
            st = len(allimgs_st2)/2.0 - 4.0*step
            end = len(allimgs_st2)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st2[ind2])
                    x[..., j+10] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass
        
        # Axial T2
        allimgs_at2 = self.get_img_paths(st_id, 'Axial T2')
        if len(allimgs_at2)==0:
            print(st_id, ': Axial T2, has no images')

        else:
            step = len(allimgs_at2) / 10.0
            st = len(allimgs_at2)/2.0 - 4.0*step
            end = len(allimgs_at2)+0.0001

            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_at2[ind2])
                    x[..., j+20] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Axial T2')
                    pass  
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, str(st_id)