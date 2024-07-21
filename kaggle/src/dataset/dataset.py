import pandas as pd
import numpy as np
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from run.config import TrainConfig, InferenceConfig

class TrainDataset(Dataset):
    def __init__(
            self, 
            cfg: TrainConfig,
            df: pd.DataFrame, 
            phase='train', 
            transform=None
        ):
        self.cfg = cfg
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        
        """
        512x512x30の3次元の配列を作成、
        in_chansはaxial,sagittal_t1,t2の3つそれぞれの画像10枚を組み合わせるための入れ物
        (512,512,0-9) -> Sagittal T1
        (512,512,10-19) -> Sagittal T2/STIR
        (512,512,20-29) -> Axial T2
        """
        x = np.zeros((
            self.cfg.dataset.image_size, 
            self.cfg.dataset.image_size, 
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
                #print(f'failed to load on {st_id}, Sagittal T1')
                pass
                
        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'{self.cfg.directory.image_dir}/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass
                
        # Axial T2
        axt2 = glob(f'{self.cfg.directory.image_dir}/{st_id}/Axial T2/*.png')
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
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass  
            
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
        ):
        self.cfg = cfg
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        
        """
        512x512x30の3次元の配列を作成、
        in_chansはaxial,sagittal_t1,t2の3つそれぞれの画像10枚を組み合わせるための入れ物
        (512,512,0-9) -> Sagittal T1
        (512,512,10-19) -> Sagittal T2/STIR
        (512,512,20-29) -> Axial T2
        """
        x = np.zeros((
            self.cfg.dataset.image_size, 
            self.cfg.dataset.image_size, 
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
                #print(f'failed to load on {st_id}, Sagittal T1')
                pass
                
        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'{self.cfg.directory.image_dir}/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass
                
        # Axial T2
        axt2 = glob(f'{self.cfg.directory.image_dir}/{st_id}/Axial T2/*.png')
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
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass  
            
        assert np.sum(x)>0
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, label