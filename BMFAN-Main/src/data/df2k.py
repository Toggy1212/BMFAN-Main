import os
from glob import glob
from data import srdata


class DF2K(srdata.SRData):
    """
    DF2K = DIV2K + Flickr2K

    This dataloader explicitly scans your existing folder structure to avoid
    naming mismatch with SRData's default scan rules.

    Expected under args.dir_data (your case: ../dataset):
      - DIV2K/DIV2K_train_HR/
      - DIV2K/DIV2K_train_LR_bicubic/X2/  (or X3/X4)
      - Flickr2K/Flickr2K_HR/
      - Flickr2K/Flickr2K_LR_bicubic/X2/  (or X3/X4)
    """

    def __init__(self, args, name='DF2K', train=True, benchmark=False):
        super(DF2K, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _set_filesystem(self, dir_data):
        # Keep parent settings (scale, rgb_range, etc.)
        super(DF2K, self)._set_filesystem(dir_data)

        # Dataset roots
        self.root_div2k = os.path.join(dir_data, 'DIV2K')
        self.root_flickr = os.path.join(dir_data, 'Flickr2K')

    @staticmethod
    def _pick_ext(ext):
        # In some repos, self.ext may be tuple like ('.png',) or ('.png', '.jpg')
        if isinstance(ext, tuple):
            return ext[0] if len(ext) > 0 else ''
        if ext is None:
            return ''
        return ext

    def _scan_one(self, root, hr_dirname, lr_dirname):
        ext = self._pick_ext(getattr(self, 'ext', '.png'))

        hr_dir = os.path.join(root, hr_dirname)
        if not os.path.isdir(hr_dir):
            return [], [[] for _ in self.scale]

        # HR list
        if ext == '':
            hr_files = sorted(glob(os.path.join(hr_dir, '*')))
        else:
            hr_files = sorted(glob(os.path.join(hr_dir, '*' + ext)))

        lr_lists = [[] for _ in self.scale]
        valid_hr = []

        missing_lr = 0
        for hr_path in hr_files:
            base = os.path.splitext(os.path.basename(hr_path))[0]

            lr_paths_this = []
            ok = True
            for s in self.scale:
                lr_path = os.path.join(root, lr_dirname, f'X{s}', f'{base}x{s}{ext}')
                if not os.path.isfile(lr_path):
                    ok = False
                    missing_lr += 1
                    break
                lr_paths_this.append(lr_path)

            if ok:
                valid_hr.append(hr_path)
                for i in range(len(self.scale)):
                    lr_lists[i].append(lr_paths_this[i])

        # Print some debug info once
        if self.train and len(hr_files) > 0:
            print(f'[{hr_dirname}] HR found: {len(hr_files)} | matched pairs: {len(valid_hr)} | missing LR: {missing_lr}')

        return valid_hr, lr_lists

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        # DIV2K
        hr1, lr1 = self._scan_one(
            self.root_div2k,
            'DIV2K_train_HR',
            'DIV2K_train_LR_bicubic'
        )
        list_hr += hr1
        for i in range(len(self.scale)):
            list_lr[i] += lr1[i]

        # Flickr2K (your folders are Flickr2K_HR / Flickr2K_LR_bicubic)
        hr2, lr2 = self._scan_one(
            self.root_flickr,
            'Flickr2K_HR',
            'Flickr2K_LR_bicubic'
        )
        list_hr += hr2
        for i in range(len(self.scale)):
            list_lr[i] += lr2[i]

        return list_hr, list_lr


