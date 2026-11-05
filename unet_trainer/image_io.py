"""Standalone Qt-independent image I/O, numeric pairing and palette labels."""
from dataclasses import dataclass
from pathlib import Path
import os,re,tempfile,json
import numpy as np
import tifffile
from PIL import Image,ImageOps

EXTENSIONS={'.tif','.tiff','.png','.jpg','.jpeg'}


def identifier(path):
    numbers=re.findall(r'\d+',Path(path).stem)
    return int(numbers[-1]) if numbers else None


@dataclass(frozen=True)
class Pair:
    number:int
    original:Path
    annotation:Path|None


def index_folder(folder):
    index={};issues=[]
    if folder is None:return index,issues
    folder=Path(folder)
    if not folder.is_dir():raise ValueError(f'Folder not found: {folder}')
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:continue
        number=identifier(p)
        if number is None:issues.append(f'No numeric ID: {p.name}');continue
        index.setdefault(number,[]).append(p)
    return index,issues


def pair_folders(originals,annotations=None):
    if annotations and Path(originals).resolve()==Path(annotations).resolve():
        raise ValueError('Choose different original and annotation folders.')
    a,issues=index_folder(originals);b,notes=index_folder(annotations);issues+=notes
    pairs=[]
    for number,paths in sorted(a.items()):
        masks=b.get(number,[])
        if len(paths)!=1 or len(masks)>1:
            issues.append(f'Ambiguous ID {number}: '+', '.join(p.name for p in paths+masks)+' — excluded');continue
        pairs.append(Pair(number,paths[0],masks[0] if masks else None))
        if not masks:issues.append(f'ID {number}: no annotation; starts with an empty mask')
    for number,paths in b.items():
        if number not in a:issues.append(f'Annotation without original, ID {number}: '+', '.join(p.name for p in paths))
    return pairs,issues


def read_image(path):
    path=Path(path)
    if path.suffix.lower() in ('.tif','.tiff'):
        with tifffile.TiffFile(path) as f:
            if len(f.pages)!=1 or len(f.series)!=1:raise ValueError('Only single-page 2D images are supported.')
            s=f.series[0]
            if s.axes not in ('YX','YXS'):raise ValueError(f'Unsupported TIFF axes: {s.axes}')
            orientation=f.pages[0].tags.get('Orientation')
            if orientation and orientation.value!=1:raise ValueError('Reorient the TIFF explicitly before import.')
            if np.prod(s.shape)>128_000_000 or np.prod(s.shape)*s.dtype.itemsize>512*1024**2:raise ValueError('Image exceeds the 512 MiB limit.')
            a=s.asarray()
    else:
        with Image.open(path) as im:
            if getattr(im,'n_frames',1)!=1:raise ValueError('Only single-frame images are supported.')
            if im.width*im.height>32_000_000:raise ValueError('Image exceeds 32 million pixels.')
            im=ImageOps.exif_transpose(im)
            if im.mode=='P':im=im.convert('RGB')
            a=np.array(im)
    if a.ndim not in (2,3) or (a.ndim==3 and a.shape[-1] not in (3,4)) or a.dtype.kind not in 'buif':raise ValueError('Expected a 2D grayscale or RGB/RGBA image.')
    if np.prod(a.shape[:2])>32_000_000 or min(a.shape)<1:raise ValueError('Image exceeds 32 million pixels or is empty.')
    return a


def read_mask(path,shape,color_map=None):
    if path is None:return np.zeros(shape,np.uint8)
    if Path(path).suffix.lower() in ('.jpg','.jpeg'):raise ValueError('JPEG masks are lossy; use a label TIFF or PNG.')
    a=read_image(path)
    if a.ndim==3:
        if a.dtype!=np.uint8:raise ValueError('Colour masks must use 8-bit RGB/RGBA samples.')
        if a.shape[-1]==4 and not np.all(a[...,3]==255):raise ValueError('Transparent annotation images are unsupported.')
        if a.shape[:2]!=shape:raise ValueError(f'Mask size {a.shape[:2]} does not match original {shape}; images are never resized.')
        rgb=a[...,:3]
        colors,inverse=np.unique(rgb.reshape(-1,3),axis=0,return_inverse=True)
        if len(colors)>65536:raise ValueError('More than 65536 distinct colours cannot be represented by uint16 labels.')
        keys=['#'+''.join(f'{int(v):02x}' for v in c) for c in colors]
        if color_map is None or any(key not in color_map for key in keys):raise ValueError('Map every annotation colour to a numeric class first.')
        values=np.asarray([color_map[key] for key in keys])
        if values.min()<0 or values.max()>65535:raise ValueError('Class values must be 0–65535.')
        a=values[inverse].reshape(shape).astype(np.uint16 if values.max()>255 else np.uint8)
    if a.ndim==2 and color_map:
        palette=indexed_palette(path)
        if palette is not None:
            ids=np.unique(a);lookup=np.arange(65536,dtype=np.uint16)
            if len({palette[int(v)] for v in ids})!=len(ids):raise ValueError('Multiple palette IDs share the same colour. Export an unambiguous numeric mask or assign distinct palette colours first.')
            for value in ids:
                key=palette[int(value)]
                if key not in color_map:raise ValueError('Map every palette colour to a class first.')
                lookup[int(value)]=color_map[key]
            a=lookup[a]
    if a.shape!=shape:raise ValueError(f'Mask size {a.shape} does not match original {shape}; images are never resized.')
    if a.dtype.kind not in 'bui' or a.min()<0 or a.max()>65535:raise ValueError('Masks must contain integer class IDs from 0 to 65535.')

    return a.astype(np.uint16 if a.max()>255 or a.dtype.itemsize>1 else np.uint8)






def save_mask(mask,path):
    path=Path(path)
    if mask.ndim!=2 or mask.dtype not in (np.dtype('uint8'),np.dtype('uint16')):raise ValueError('Expected uint8/uint16 label mask.')
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,suffix='.tmp',dir=path.parent)
    os.close(fd)
    try:
        with open(tmp,'wb') as stream:
            tifffile.imwrite(stream,mask,metadata={'axes':'YX'});stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def mask_colors(path):
    a=read_image(path)
    if a.ndim!=3:return []
    if a.dtype!=np.uint8:raise ValueError('Colour masks must be 8-bit RGB/RGBA.')
    if a.shape[-1]==4 and not np.all(a[...,3]==255):raise ValueError('Transparent annotation masks are unsupported.')
    colors=np.unique(a[...,:3].reshape(-1,3),axis=0)
    if len(colors)>65536:raise ValueError('More than 65536 colours exceeds the uint16 label capacity.')
    return ['#'+''.join(f'{int(v):02x}' for v in c) for c in colors]




def load_labels(path):
    record=json.loads(Path(path).read_text())
    if record.get('schema')!=1:raise ValueError('Unknown class-label format.')
    definitions={int(k):v for k,v in record['classes'].items()};mapping=record.get('input_color_to_label',{})
    if len(definitions)>65536 or any(not 0<=k<=65535 or not isinstance(v.get('name'),str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',v.get('color','')) for k,v in definitions.items()):raise ValueError('Invalid class definitions.')
    if any(not re.fullmatch(r'#[0-9a-f]{6}',k) or v not in definitions for k,v in mapping.items()):raise ValueError('Invalid colour mapping.')
    if record.get('background_id',0) not in definitions:raise ValueError('Background class is missing from definitions.')
    return definitions,mapping


def indexed_palette(path):
    path=Path(path)
    if path.suffix.lower() in ('.tif','.tiff'):
        with tifffile.TiffFile(path) as f:
            palette=f.pages[0].colormap
            if palette is None:return None
            # ImageJ palettes use 8-bit components shifted left by 8; other TIFFs
            # can span the full unsigned 16-bit range.
            divisor=256 if np.all(palette%256==0) else 257
            rgb=np.rint(palette.T.astype(float)/divisor).clip(0,255).astype(np.uint8)
            return ['#'+''.join(f'{int(v):02x}' for v in c) for c in rgb]
    return None


def mask_inventory(path):
    a=read_image(path);palette=indexed_palette(path)
    if a.ndim==2:
        ids,counts=np.unique(a,return_counts=True)
        if a.dtype.kind not in 'bui' or ids.min()<0 or ids.max()>65535:raise ValueError('Expected integer class labels 0–65535.')
        return [{'color':palette[int(v)] if palette else None,'source_id':int(v),'count':int(n)} for v,n in zip(ids,counts)]
    colors=mask_colors(path)
    _,counts=np.unique(a[...,:3].reshape(-1,3),axis=0,return_counts=True)
    return [{'color':color,'source_id':None,'count':int(n)} for color,n in zip(colors,counts)]
