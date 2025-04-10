import torch
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import codecs as cs
import dataset.util.plot as plot_util
from dataset.util.humanml3d.util.paramUtil import *
import dataset.humanml3d_dataset as humanml3d_dataset
import json
from numpy import dot
from numpy.linalg import norm

class HumanML3D(humanml3d_dataset.HumanML3D):
    NAME = 'HumanML3D_TEXT'
    def __init__(self, config):
        # Initialize attributes needed for parent class initialization
        self.labels = []
        self.text_path = config['data']['text_path']
        self.text_model_path = config['data']['text_model_path']
        
        # Call parent class initialization first
        super().__init__(config)
        
        # Initialize text model after parent class initialization
        self.sentence_transformer = self.init_text_model(self.text_model_path)
        
        # Load embeddings
        self.train_embs = self.load_sentence_embedding_from_split(self.train_split_file)
        #self.val_embs = self.load_sentence_embedding_from_split(self.val_split_file)
        #self.test_embs = self.load_sentence_embedding_from_split(self.test_split_file)

    def init_text_model(self, path):
        from sentence_transformers import SentenceTransformer
        
        # 모델 디렉토리 경로 확인
        model_dir = os.path.dirname(path)
        if not os.path.exists(model_dir):
            print(f"Creating directory: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
        
        # 모델 존재 여부 확인
        if os.path.exists(path):
            print(f'Loading local model: {path}')
            model = SentenceTransformer(path)
        else:
            # 모델이 없으면 허깅페이스에서 다운로드
            print(f'Path({path}) not found, downloading from Hugging Face...')
            # 기본 모델 사용
            fallback_model = "all-MiniLM-L6-v2"
            print(f'Using fallback model: {fallback_model}')
            model = SentenceTransformer(fallback_model)
            # 모델 저장
            try:
                model.save(path)
                print(f'Model saved to {path}')
            except Exception as e:
                print(f"Failed to save model: {e}")
                print("Continuing with in-memory model")
        
        return model

    def encode_text(self, text):
        return self.sentence_transformer.encode(text)

    def load_sentence_embedding_from_split(self, split_file, reprocess=False):
        base_name = os.path.basename(split_file)[:-4]
        base_dir = os.path.dirname(split_file)
        out_emb_file = os.path.join(base_dir, base_name+'_emb.pt')
        out_dict_file = os.path.join(base_dir, base_name+'_dict')

        if reprocess or not os.path.exists(out_emb_file):
            text_dicts = {}
            text_name_lst = []
            texts = []
            
            with open(split_file) as f:
                lines = [osp.join(self.text_path,x.strip()+'.txt') for x in f.readlines()]
            text_idx = 0
            
            for i, line in enumerate(tqdm(lines)):
                data_lst = self.process_text(line)
                for cur_dict in data_lst:
                    motion_name = cur_dict['motion_name']
                    if motion_name not in text_dicts:
                        text_dicts[motion_name] = []
                    text_name_lst.append(motion_name)
                    cur_dict['text_idx'] = text_idx
                    text_dicts[motion_name].append(cur_dict)
                    texts.append(cur_dict['caption'])
                    text_idx += 1

            embs = self.encode_text(texts)
            torch.save(embs, out_emb_file)
            with open(out_dict_file, 'w') as fout:
                json.dump(text_dicts, fout)

        else:
            try:
                # PyTorch 2.6에서 weights_only 기본값이 True로 변경되어 명시적으로 False 지정
                embs = torch.load(out_emb_file, weights_only=False)
                with open(out_dict_file, 'r') as fin:
                    text_dicts = json.load(fin)
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                print("Regenerating embeddings...")
                return self.load_sentence_embedding_from_split(split_file, reprocess=True)
        return embs, text_dicts

    
    def load_new_dataset(self, split):
        new_data = []
        with open(split) as f:
            lines = [osp.join(self.path,x.strip()+'.npy') for x in f.readlines()]
        
        for i, line in enumerate(tqdm(lines)):
            data = self.process_data(line)
            #data = self.load_new_data(line)
            #data = self.transform_new_data(data)
            new_data.append(data)

        #new_data_flattened = np.array(new_data_flattened)
        return new_data


    def process_text(self,fname):
        text_data = []
        try:
            # utf-8 인코딩으로 파일 열기
            with open(fname, 'r', encoding='utf-8') as f:
                file_base_name = os.path.basename(fname)
                for i, line in enumerate(f.readlines()):
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['motion_name'] = file_base_name
                    text_dict['motion_idx'] = i
                    text_dict['tokens'] = tokens
                    text_dict['st_frame'] = int(f_tag) * 20
                    text_dict['end_frame'] = int(to_tag) * 20
                    text_data.append(text_dict)
        except Exception as e:
            print(f"Error processing file {fname}: {e}")
        return text_data
        

