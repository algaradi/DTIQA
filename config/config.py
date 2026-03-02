import os

class Config:
    def __init__(self):
        # ---------------------------------------------------------
        # Dataset Paths
        # ---------------------------------------------------------
        # These paths point to the dataset directories. 
        # Modify these if evaluating on a local machine.
        self.folder_path = {
            'live': '/root/autodl-tmp/LIVEIQA_release2/',
            'csiq': '/root/autodl-tmp/CSIQ/',
            'tid2013': '/root/autodl-tmp/TID2013/',
            'livec': '/root/autodl-tmp/live_challenge/',
            'koniq-10k': '/root/autodl-tmp/koniq-10k/',
            'bid': '/root/autodl-tmp/BID/',
        }

        # ---------------------------------------------------------
        # Dataset Image Indices (Subset bounds)
        # ---------------------------------------------------------
        self.img_num = {
            'live': list(range(0, 29)),
            'csiq': list(range(0, 30)),
            'tid2013': list(range(0, 25)),
            'livec': list(range(0, 1162)),
            'koniq-10k': list(range(0, 10073)),
            'bid': list(range(0, 586)),
        }

        # ---------------------------------------------------------
        # Per-Distortion Evaluation Subsets
        # ---------------------------------------------------------
        # The indices belonging to specific distortion types for evaluation.
        self.distortion_indices = {
            'live': {
                'jp2k': list(range(0, 227)),
                'jpeg': list(range(227, 460)),
                'wn':   list(range(460, 634)),
                'gblur': list(range(634, 808)),
                'ff':   list(range(808, 982)),
            },
            'csiq': {
                'awgn':  list(range(0, 150)),
                'jpeg':  list(range(150, 300)),
                'jpeg2000': list(range(300, 450)),
                'fnoise': list(range(450, 600)),
                'blur':  list(range(600, 750)),
                'contrast': list(range(750, 866)),
            }
        }

# Global singleton instance for easy import
cfg = Config()
