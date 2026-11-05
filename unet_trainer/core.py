"""Qt-free data, U-Net, training, metrics and checkpoint inference."""
from pathlib import Path
import json, math, time, os, re
import numpy as np
import torch
from torch import nn
from .image_io import EXTENSIONS, pair_folders, mask_inventory, read_image, read_mask, save_mask


def devices():
    return ['cpu'] + (['cuda'] if torch.cuda.is_available() else []) + (['mps'] if torch.backends.mps.is_available() else [])


def inspect_dataset(originals, masks):
    pairs, issues = pair_folders(originals, masks)
    # Ambiguous or missing pairs must be resolved, never silently selected.
    if issues: raise ValueError('\n'.join(issues))
    if not pairs: raise ValueError('No numbered image/mask pairs found.')
    definitions = {}; rgb = {}; channels = None
    for pair in pairs:
        source = read_image(pair.original)
        if source.ndim == 3 and source.shape[2] == 4: raise ValueError('Use grayscale or RGB originals, without alpha.')
        count = 1 if source.ndim == 2 else 3
        if channels is not None and channels != count: raise ValueError('All originals must have the same channel count.')
        channels = count
        if not np.isfinite(source).all(): raise ValueError(f'Nonfinite pixels in {pair.original.name}.')
        mask = read_image(pair.annotation)
        if mask.shape[:2] != source.shape[:2]: raise ValueError(f'Shape mismatch for ID {pair.number}; no automatic resizing.')
        for row in mask_inventory(pair.annotation):
            ident, color = row['source_id'], row['color']
            if ident is None:
                rgb[color] = rgb.get(color, 0) + row['count']; continue
            if ident in definitions and color and definitions[ident]['color'] != color:
                raise ValueError(f'Inconsistent palette colour for class {ident}. Standardize masks first.')
            entry = definitions.setdefault(ident, {'id': ident, 'name': f'Class {ident}', 'color': color or '#%06x' % ((ident * 2654435761) % 0xffffff), 'count': 0})
            entry['count'] += row['count']
    for color, count in sorted(rgb.items()):
        matches = [k for k,v in definitions.items() if v['color'] == color]
        ident = matches[0] if matches else next(i for i in range(65536) if i not in definitions)
        entry = definitions.setdefault(ident, {'id':ident, 'name':f'Class {ident}', 'color':color, 'count':0})
        entry['count'] += count
    if not 2 <= len(definitions) <= 256: raise ValueError('Training supports 2–256 classes; annotation supports up to 65536 IDs.')
    colors = [d['color'] for d in definitions.values()]
    if len(colors) != len(set(colors)): raise ValueError('Different IDs have identical colours; assign distinct colours before training.')
    return {'pairs':[{'number':p.number,'original':str(p.original),'mask':str(p.annotation)} for p in pairs], 'classes':list(sorted(definitions.values(),key=lambda d:d['id'])), 'channels':channels}


def normalize(image):
    a = image.astype(np.float32)
    if not np.isfinite(a).all(): raise ValueError('Nonfinite source pixels are unsupported.')
    if a.ndim == 2: a = a[...,None]
    if a.shape[-1] not in (1,3): raise ValueError('Use grayscale or RGB images.')
    lo, hi = np.percentile(a, [1,99], axis=(0,1), keepdims=True)
    return np.moveaxis(np.clip((a-lo)/np.maximum(hi-lo,1e-8),0,1).astype(np.float32), -1, 0)


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(1,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(1,b),nn.ReLU())


class UNet(nn.Module):
    def __init__(self, channels, classes, base=16, depth=3):
        super().__init__()
        widths = [base*2**i for i in range(depth+1)]
        self.down = nn.ModuleList([Block(channels if i==0 else widths[i-1], w) for i,w in enumerate(widths)])
        self.up = nn.ModuleList([Block(widths[i+1]+widths[i], widths[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(base,classes,1)
    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.down):
            if i: x = nn.functional.max_pool2d(x,2)
            x = layer(x); skips.append(x)
        for layer, skip in zip(self.up,reversed(skips[:-1])):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = layer(torch.cat([x,skip],1))
        return self.out(x)


def scores(confusion):
    tp = np.diag(confusion).astype(float); actual=confusion.sum(1); predicted=confusion.sum(0)
    union=actual+predicted-tp; denom=actual+predicted
    iou=np.divide(tp,union,out=np.full_like(tp,np.nan),where=union>0)
    dice=np.divide(2*tp,denom,out=np.full_like(tp,np.nan),where=denom>0)
    return {'iou':[float(v) if np.isfinite(v) else None for v in iou], 'dice':[float(v) if np.isfinite(v) else None for v in dice], 'miou':float(np.nanmean(iou)), 'mean_dice':float(np.nanmean(dice))}


def loss_value(logits, target, kind):
    ce=nn.functional.cross_entropy(logits,target)
    if kind=='Cross-entropy': return ce
    probs=logits.softmax(1); one=nn.functional.one_hot(target,logits.shape[1]).permute(0,3,1,2)
    dice=(2*(probs*one).sum((0,2,3))+1e-6)/(probs.sum((0,2,3))+one.sum((0,2,3))+1e-6)
    return ce + 1-dice.mean()


def padded(a, size):
    return np.pad(a,[(0,0)]*(a.ndim-2)+[(0,max(0,size-a.shape[-2])),(0,max(0,size-a.shape[-1]))],mode='edge')


def infer(model, image, device, patch):
    """Overlapping half-stride tiles, average probabilities, retain native dimensions."""
    c=model.out.out_channels; h,w=image.shape[-2:]
    if (c+1)*max(h,patch)*max(w,patch)*4 > 512*1024**2: raise ValueError('Probability map exceeds 512 MiB; crop the source explicitly.')
    a=padded(image,patch); hh,ww=a.shape[-2:]; total=np.zeros((c,hh,ww),np.float32); counts=np.zeros((hh,ww),np.float32)
    ys=sorted(set(list(range(0,hh-patch+1,max(1,patch//2)))+[hh-patch])); xs=sorted(set(list(range(0,ww-patch+1,max(1,patch//2)))+[ww-patch]))
    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                p=model(torch.from_numpy(a[:,y:y+patch,x:x+patch][None]).to(device)).softmax(1)[0].cpu().numpy()
                total[:,y:y+patch,x:x+patch]+=p; counts[y:y+patch,x:x+patch]+=1
    return total[:,:h,:w]/counts[None,:h,:w]


def atomic_checkpoint(record,path):
    path=Path(path); tmp=path.with_suffix('.tmp')
    try:
        torch.save(record,tmp); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def _checkpoint_record(checkpoint):
    """Safely load and validate a trainer checkpoint."""
    record=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(record,dict) or record.get('schema')!=1 or record.get('normalization')!='per-image-channel-percentile-1-99':
        raise ValueError('Unsupported U-Net checkpoint.')
    architecture=record.get('architecture')
    classes=record.get('classes')
    patch=record.get('patch')
    if not isinstance(architecture,dict) or set(architecture)!={'channels','classes','base','depth'}:
        raise ValueError('Classifier architecture metadata is invalid.')
    if architecture['channels'] not in (1,3) or not isinstance(architecture['classes'],int) or architecture['classes']<2:
        raise ValueError('Classifier architecture metadata is invalid.')
    if not isinstance(classes,list) or len(classes)!=architecture['classes']:
        raise ValueError('Classifier class metadata is invalid.')
    seen=set()
    for item in classes:
        if not isinstance(item,dict) or not isinstance(item.get('id'),int) or not 0<=item['id']<=65535 or item['id'] in seen:
            raise ValueError('Classifier class metadata is invalid.')
        if not isinstance(item.get('name'),str) or not item['name'].strip() or not isinstance(item.get('color'),str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',item['color']):
            raise ValueError('Classifier class metadata is invalid.')
        seen.add(item['id'])
    if not isinstance(patch,int) or patch<1:
        raise ValueError('Classifier patch metadata is invalid.')
    return record


def checkpoint_metadata(checkpoint):
    """Return validated classifier details suitable for the GUI."""
    record=_checkpoint_record(checkpoint)
    architecture=record['architecture'];classes=record['classes'];patch=record['patch']
    return {'architecture':architecture,'classes':classes,'patch':patch,'epoch':record.get('epoch'),'metrics':record.get('metrics',{})}


def train(config, emit=lambda event:None):
    torch.set_num_threads(2)
    device=config['device']
    if device not in devices(): raise ValueError(f'Requested device {device} is unavailable; no fallback.')
    ds=config['dataset']; pairs=ds['pairs']; classes=ds['classes']; n=len(classes)
    if len(pairs)<2: raise ValueError('At least two annotated images are required for an image-level validation split. One image can be inspected but is insufficient to train with validation.')
    patch=int(config['patch']); depth=int(config['depth']);base=int(config['base'])
    if patch<2**depth or patch>1024 or base not in (4,8,16,32,64) or not 1<=depth<=4: raise ValueError('Invalid U-Net dimensions.')
    if not 0<config['validation']<1 or config['epochs']<1 or config['batch']<1 or config['patches']<1:raise ValueError('Invalid training settings.')
    seed=int(config['seed']);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    order=rng.permutation(len(pairs)); nv=max(1,min(len(pairs)-1,round(len(pairs)*config['validation'])))
    val_ids=order[:nv].tolist();train_ids=order[nv:].tolist()
    mapping={d['color']:d['id'] for d in classes};ids=[d['id'] for d in classes];data=[];memory=0
    lookup=np.full(65536,-1,np.int64);lookup[ids]=np.arange(n)
    for pair in pairs:
        image=normalize(read_image(pair['original'])); mask=read_mask(pair['mask'],image.shape[1:],mapping); label=lookup[mask]
        if (label<0).any():raise ValueError('Mask has an undefined class ID.')
        memory+=image.nbytes+label.nbytes
        if memory>512*1024**2:raise ValueError('Dataset exceeds 512 MiB in-memory training limit; use a smaller selection.')
        data.append((image,label))
    present=set(np.unique(np.concatenate([np.unique(data[i][1]) for i in train_ids])).tolist())
    if len(present)!=n:raise ValueError('Some classes are absent from training images. Change seed/split or provide more examples.')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'run.json').write_text(json.dumps({**config,'training_indices':train_ids,'validation_indices':val_ids},indent=2))
    model=UNet(ds['channels'],n,base,depth).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    best=-1.;stale=0;history=[]
    for epoch in range(1,config['epochs']+1):
        started=time.monotonic();model.train();train_loss=0.;seen=0
        for offset in range(0,config['patches'],config['batch']):
            images=[];labels=[]
            for _ in range(min(config['batch'],config['patches']-offset)):
                im,mask=data[int(rng.choice(train_ids))];im=padded(im,patch);mask=padded(mask,patch)
                y=int(rng.integers(im.shape[-2]-patch+1));x=int(rng.integers(im.shape[-1]-patch+1))
                im=im[:,y:y+patch,x:x+patch];mask=mask[y:y+patch,x:x+patch]
                if rng.random()<.5:im=im[:,:,::-1];mask=mask[:,::-1]
                if rng.random()<.5:im=im[:,::-1,:];mask=mask[::-1,:]
                images.append(im.copy());labels.append(mask.copy())
            inputs=torch.from_numpy(np.stack(images)).to(device);targets=torch.from_numpy(np.stack(labels)).to(device)
            optimizer.zero_grad();loss=loss_value(model(inputs),targets,config['loss']);loss.backward();optimizer.step()
            train_loss+=float(loss.detach().cpu())*len(images);seen+=len(images)
        confusion=np.zeros((n,n),np.int64);val_loss=0.;pixels=0
        for j in val_ids:
            im,truth=data[j];prob=infer(model,im,device,patch);pred=prob.argmax(0)
            confusion+=np.bincount((truth*n+pred).ravel(),minlength=n*n).reshape(n,n)
            # Full-image CE, and optional soft Dice, using averaged tile probabilities.
            ce=-np.log(np.maximum(np.take_along_axis(prob,truth[None],axis=0),1e-7)).mean()
            soft=[]
            for k in range(n):
                target=truth==k;soft.append((2*prob[k][target].sum()+1e-6)/(prob[k].sum()+target.sum()+1e-6))
            val_loss+=(ce+(1-np.mean(soft) if config['loss']!='Cross-entropy' else 0))*truth.size;pixels+=truth.size
            if j==val_ids[0]:
                np.savez_compressed(out/'preview.tmp.npz',image=np.moveaxis(im,0,-1),truth=truth,prediction=pred)
                os.replace(out/'preview.tmp.npz',out/'preview.npz')
        metrics=scores(confusion);seconds=time.monotonic()-started
        row={'epoch':epoch,'train_loss':train_loss/seen,'val_loss':float(val_loss/pixels),**metrics,'seconds':seconds,'remaining':seconds*(config['epochs']-epoch)}
        history.append(row)
        if metrics['miou']>best:
            best=metrics['miou'];stale=0
            record={'schema':1,'architecture':{'channels':ds['channels'],'classes':n,'base':base,'depth':depth},'classes':classes,'patch':patch,'normalization':'per-image-channel-percentile-1-99','epoch':epoch,'metrics':metrics,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            atomic_checkpoint(record,out/'best.unet.pt')
        else:stale+=1
        (out/'history.json').write_text(json.dumps(history,indent=2));emit({'type':'epoch',**row,'best':best,'output':str(out)})
        if config['patience'] and stale>=config['patience']:break
    emit({'type':'done','output':str(out),'checkpoint':str(out/'best.unet.pt')})
    return out/'best.unet.pt'


def _prediction_model(checkpoint, device):
    if device not in devices():raise ValueError('Selected device unavailable.')
    # weights_only disallows arbitrary executable pickle objects.
    record=_checkpoint_record(checkpoint)
    model=UNet(**record['architecture']);model.load_state_dict(record['state_dict']);model.to(device)
    return record,model


def _predict_loaded(record, model, checkpoint, source, device, output):
    image=normalize(read_image(source))
    if image.shape[0]!=record['architecture']['channels']:raise ValueError('Image channel count does not match classifier.')
    probabilities=infer(model,image,device,record['patch'])
    ids=np.asarray([c['id'] for c in record['classes']],dtype=np.uint16)
    mask=ids[probabilities.argmax(0)]
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    save_mask(mask,out/'prediction.tif');np.savez_compressed(out/'probabilities.npz',probabilities=probabilities,class_ids=ids)
    (out/'classes.json').write_text(json.dumps(record['classes'],indent=2))
    (out/'source.json').write_text(json.dumps({'image':str(source),'checkpoint':str(checkpoint),'device':device},indent=2))
    return out


def predict(checkpoint, source, device, output):
    record,model=_prediction_model(checkpoint,device)
    return _predict_loaded(record,model,checkpoint,source,device,output)


def predict_directory(checkpoint, source_directory, device, output, emit=None):
    source_directory=Path(source_directory).expanduser().resolve()
    if not source_directory.is_dir():raise ValueError('Input image directory not found: '+str(source_directory))
    sources=sorted((path for path in source_directory.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS),key=lambda path:path.name.lower())
    if not sources:raise ValueError('The input image directory contains no supported TIFF, PNG or JPEG images.')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record,model=_prediction_model(checkpoint,device);items=[]
    for index,source in enumerate(sources,1):
        stem=re.sub(r'[^A-Za-z0-9._-]+','-',source.stem).strip('._-') or 'image'
        result=_predict_loaded(record,model,checkpoint,source,device,out/f'{index:04d}-{stem}')
        item={'source':str(source),'output':str(result)};items.append(item)
        if emit:emit({'type':'batch_prediction',**item,'index':index,'total':len(sources)})
    manifest={'schema':1,'input_directory':str(source_directory),'checkpoint':str(checkpoint),'device':device,'images':items}
    (out/'batch.json').write_text(json.dumps(manifest,indent=2))
    return out
