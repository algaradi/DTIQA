# DTIQA Datasets

This directory manages the data instructions for the DTIQA framework.

Due to file sizes, the datasets themselves are not included in this repository. **You must download all datasets directly from their official distribution sources.**

### ⚠️ Critical Instruction: Preserve Official Folder Structures

When downloading the datasets from their official websites, **do not** alter, reorganize, or rename any of the internal folders (e.g., `fastfading/`, `gblur/`, `jpeg/`, etc.).

Our custom PyTorch DataLoader implementation specifically expects and automatically parses the exact, original folder hierarchy provided by the dataset publishers to extract distortion indices.

> [!IMPORTANT]
> If you change the official folder structures after downloading, the dataset loaders will fail to map the distortion types correctly.

---

## 🔗 Dataset Links

We have provided specific `download.md` instruction files located within each dataset's respective subfolder to help you find the correct official downloads:

### Synthetic Datasets

* `LIVE/` (LIVE Image Quality Assessment Database Release 2)
* `CSIQ/` (Categorical Image Quality Database)
* `TID2013/` (Tampere Image Database 2013)

### Authentic Datasets

* `LIVEC/` (LIVE In the Wild Image Quality Challenge Database)
* `KonIQ-10k/` (Konstanz Artificially Distorted Image Database)
* `BID/` (Blur Image Database)

---

## ⚙️ Setting Up Configuration Paths

Once you have successfully downloaded and extracted the official datasets, you must bind them to our framework.

Open the **`config/config.py`** file and update the `self.folder_path` dictionary with the absolute directory paths pointing to where you extracted each dataset on your machine:

```python
self.folder_path = {
    'live': '/path/to/officially/extracted/LIVEIQA_release2/',
    'csiq': '/path/to/officially/extracted/csiq/',
    'tid2013': '/path/to/officially/extracted/TID2013/',
    'livec': '/path/to/officially/extracted/live_challenge/',
    'koniq-10k': '/path/to/officially/extracted/koniq-10k/',
    'bid': '/path/to/officially/extracted/BID/',
}
```

The DTIQA training and evaluation pipelines will automatically route to these locations!
