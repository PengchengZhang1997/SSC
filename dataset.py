import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
import itertools
from PIL import Image
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.

        """
        print(opt.batch_size)
        self.opt = opt
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)

            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=False, drop_last=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self, meta_target_index=-1, no_pseudo=False):
        balanced_batch_images = []
        balanced_batch_texts = []
        balanced_domain_ids = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            if i == meta_target_index: continue
            if i == len(self.dataloader_iter_list) - 1 and no_pseudo and self.has_pseudo_label_dataset(): continue 
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
                balanced_domain_ids += [i] * len(text)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
                balanced_domain_ids += [i] * len(text)
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        balanced_domain_ids = torch.tensor(balanced_domain_ids)

        return balanced_batch_images, balanced_batch_texts, balanced_domain_ids
    
    def get_meta_test_batch(self, meta_target_index=-1):
        
        if meta_target_index == self.opt.source_num:
            assert len(self.data_loader_list) == self.opt.source_num + 1, 'There is no target dataset'
        balanced_batch_images = []
        balanced_batch_texts = []
        balanced_domain_ids = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            if i == meta_target_index:
                try:
                    image, text = data_loader_iter.next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                    balanced_domain_ids += [i] * len(text)
                except StopIteration:
                    self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                    image, text = self.dataloader_iter_list[i].next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                    balanced_domain_ids += [i] * len(text)
                except ValueError:
                    pass
        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        balanced_domain_ids = torch.tensor(balanced_domain_ids)

        return balanced_batch_images, balanced_batch_texts, balanced_domain_ids

    def add_target_domain_dataset(self, dataset, opt):
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=_AlignCollate, drop_last=True)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def add_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def add_residual_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_residual_pseudo_label_dataset():
            self.data_loader_list[opt.source_num + 1] = self_training_loader
            self.dataloader_iter_list[opt.source_num + 1] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def has_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num else False

    def has_residual_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num + 1 else False
        
class Batch_Balanced_Sampler(object):
    def __init__(self, dataset_len, batch_size):
        dataset_len.insert(0,0)
        self.dataset_len = dataset_len
        self.start_index = list(itertools.accumulate(self.dataset_len))[:-1]
        self.batch_size = batch_size
        self.counter = 0

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        data_index = []
        while True:
            for i in range(len(self.start_index)):
                data_index.extend([self.start_index[i] + (self.counter * self.batch_size + j) % self.dataset_len[i + 1] for j in range(self.batch_size)])
            yield data_index
            data_index = []
            self.counter += 1
        

class Batch_Balanced_Dataset0(object):

    def __init__(self, opt):
        self.opt = opt
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.batch_size_list = []
        Total_batch_size = 0

        self.dataset_list = []
        self.dataset_len_list = []

        self.pseudo_dataloader = None
        self.pseudo_batch_size = -1

        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            if opt.fix_dataset_num != -1: number_dataset = opt.fix_dataset_num
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)

            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            self.batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            self.dataset_list.append(_dataset)
            self.dataset_len_list.append(number_dataset)



        concatenated_dataset = ConcatDataset(self.dataset_list)
        assert len(concatenated_dataset) == sum(self.dataset_len_list)

        batch_sampler = Batch_Balanced_Sampler(self.dataset_len_list, _batch_size)
        self.data_loader = iter(torch.utils.data.DataLoader(
            concatenated_dataset,
            batch_sampler=batch_sampler,
            num_workers=int(opt.workers),
            collate_fn=_AlignCollate, pin_memory=False))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(self.batch_size_list)
        self.batch_size_list = list(map(int, self.batch_size_list))
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self, meta_target_index=-1, no_pseudo=False):
        
        imgs, texts = next(self.data_loader)
        if meta_target_index == -1 or meta_target_index >= len(self.batch_size_list): return imgs, texts
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)

        ret_imgs, ret_texts = [], []
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index: continue
            ret_imgs.extend(imgs[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
            ret_texts.extend(texts[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
        ret_imgs = torch.stack(ret_imgs, 0)

        if self.has_pseudo_label_dataset():
            try:
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            ret_imgs = torch.cat([ret_imgs, psuedo_imgs], 0)
            ret_texts += pseudo_texts

        return ret_imgs, ret_texts

    def get_meta_test_batch(self, meta_target_index=-1):
        
        assert meta_target_index != -1, 'Meta target index should be specified'
        if meta_target_index >= len(self.batch_size_list) and self.has_pseudo_label_dataset(): 
            try:
                img, text = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                img, text = next(self.pseudo_dataloader_iter)

            return img, text
        
        imgs, texts = next(self.data_loader)
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)
        ret_img, ret_text = None, None
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index:
                ret_img = imgs[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]
                ret_text = texts[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]

        return ret_img, ret_text

    def add_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self.pseudo_batch_size = batch_size
        self.pseudo_dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)


    def has_pseudo_label_dataset(self):
        return True if self.pseudo_dataloader else False


def hierarchical_dataset(root, opt, select_data='/', pseudo=False):
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/', followlinks=True):
        print(dirpath, dirnames, filenames)
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        continue

                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def self_training_collate(batch):
    imgs, labels = [], []
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
    
    return torch.stack(imgs), labels

class SelfTrainingDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        assert len(self.imgs) == len(self.labels)
        return len(self.imgs)



def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
