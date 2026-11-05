import json
from pathlib import Path
import numpy as np
import pytest
import tifffile
from unet_trainer.core import checkpoint_metadata,inspect_dataset,train,predict,normalize,scores,UNet,infer,devices


def dataset(tmp_path):
    a=tmp_path/'originals';b=tmp_path/'masks';a.mkdir();b.mkdir()
    for i in range(3):
        y,x=np.mgrid[:24,:28];mask=((x+y+i)>25).astype(np.uint8)
        image=(mask.astype(np.uint16)*40000+x*100+y*10).astype(np.uint16)
        tifffile.imwrite(a/f'image-{i:03d}.tif',image);tifffile.imwrite(b/f'Annotated_{i}.tif',mask)
    return inspect_dataset(a,b)


def config(tmp_path,ds,device='cpu'):
    return dict(dataset=ds,device=device,output=str(tmp_path/'run'),patch=16,base=4,depth=2,epochs=2,batch=2,patches=4,validation=.34,seed=42,lr=.001,weight_decay=.0001,loss='Cross-entropy + Dice',patience=2)


def test_train_reload_predict_and_metrics(tmp_path):
    ds=dataset(tmp_path);before=Path(ds['pairs'][0]['original']).read_bytes();events=[]
    checkpoint=train(config(tmp_path,ds),events.append)
    assert checkpoint.exists() and events[-1]['type']=='done'
    run=json.loads((tmp_path/'run/run.json').read_text())
    assert not set(run['training_indices']) & set(run['validation_indices'])
    assert len(events[0]['iou'])==2 and 0<=events[0]['miou']<=1
    out=predict(checkpoint,ds['pairs'][0]['original'],'cpu',tmp_path/'prediction')
    with np.load(out/'probabilities.npz') as f:
        assert f['probabilities'].shape==(2,24,28);np.testing.assert_allclose(f['probabilities'].sum(0),1,atol=1e-6)
    assert tifffile.imread(out/'prediction.tif').shape==(24,28)
    assert Path(ds['pairs'][0]['original']).read_bytes()==before
    with pytest.raises(FileExistsError):predict(checkpoint,ds['pairs'][0]['original'],'cpu',out)
    result=scores(np.array([[2,1],[0,3]]));assert result['iou']==[2/3,3/4]
    assert normalize(np.ones((7,9),np.uint16)).shape==(1,7,9)
    # Odd native sizes retained through padding and tiling.
    probability=infer(UNet(1,2,4,2),np.ones((1,7,9),np.float32),'cpu',16)
    assert probability.shape==(2,7,9)


def test_dataset_rejections_and_palette(tmp_path):
    ds=dataset(tmp_path)
    cfg=config(tmp_path,{**ds,'pairs':ds['pairs'][:1]})
    with pytest.raises(ValueError,match='At least two'):train(cfg)
    cfg=config(tmp_path,ds,'not-a-device')
    with pytest.raises(ValueError,match='unavailable'):train(cfg)
    a=tmp_path/'originals';b=tmp_path/'masks'
    (b/'Annotated_0.tif').unlink()
    with pytest.raises(ValueError,match='no annotation'):inspect_dataset(a,b)
    palette=np.zeros((3,256),np.uint16);palette[:,0]=[65280,0,0];palette[:,4]=[21504,57856,65280]
    for i in range(3):
        mask=np.zeros((24,28),np.uint8);mask[:,12:]=4;tifffile.imwrite(b/f'Annotated_{i}.tif',mask,photometric='palette',colormap=palette)
    ds=inspect_dataset(a,b);assert [c['id'] for c in ds['classes']]==[0,4]
    assert ds['classes'][1]['color']=='#54e2ff'
    tifffile.imwrite(b/'Annotated_2.tif',np.zeros((10,10),np.uint8))
    with pytest.raises(ValueError,match='Shape mismatch'):inspect_dataset(a,b)


def test_negative_mask_values_are_rejected(tmp_path):
    from unet_trainer.image_io import mask_inventory,read_mask
    path=tmp_path/'negative.tif';tifffile.imwrite(path,np.array([[-1,1]],dtype=np.int16))
    with pytest.raises(ValueError,match='0 to 65535'):read_mask(path,(1,2))
    with pytest.raises(ValueError,match='0–65535'):mask_inventory(path)


def test_native_dashboard_worker_train_and_cancel(qtbot,tmp_path):
    from unet_trainer.window import TrainingWindow
    w=TrainingWindow();qtbot.addWidget(w);w.show()
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())]==['Train classifier','Apply classifier']
    assert not w.paths['output'].text() and not w.apply_output.text()
    for combo in (w.device,w.apply_device):
        cuda=combo.findData('cuda');assert cuda>=0
        assert combo.model().item(cuda).isEnabled()==('cuda' in devices())
    ds=dataset(tmp_path)
    for key in ('originals','masks'):w.paths[key].setText(str(tmp_path/key))
    w.paths['output'].setText(str(tmp_path/'results'));assert not w.start.isEnabled()
    w.load_data();qtbot.waitUntil(lambda:w.process is None,timeout=30000)
    assert len(w.dataset['classes'])==2 and not w.start.isEnabled()
    w.table.item(0,2).setText('Matrix');w.table.item(1,2).setText('Particle')
    w.device.setCurrentIndex(w.device.findData('cpu'));assert w.start.isEnabled()
    for key,value in dict(epochs=1,batch=1,patch=16,patches=2,depth=2).items():w.settings[key].setValue(value)
    w.base.setCurrentText('4');w.start_training();assert not w.controls.isEnabled()
    qtbot.waitUntil(lambda:w.process is None,timeout=60000)
    assert len(w.history)==1 and Path(w.checkpoint.text()).is_file()
    assert w.job_classes[0]['name']=='Matrix' and not w.previews[0].pixmap().isNull()
    w.settings['epochs'].setValue(100);w.settings['patches'].setValue(1000);w.start_training();w.cancel_job()
    qtbot.waitUntil(lambda:w.process is None,timeout=15000);assert w.controls.isEnabled()
    w.close()


def test_apply_tab_loads_classifier_and_shows_side_by_side_result(qtbot,tmp_path):
    from unet_trainer.window import TrainingWindow
    ds=dataset(tmp_path);checkpoint=train(config(tmp_path,ds))
    metadata=checkpoint_metadata(checkpoint);assert metadata['architecture']['classes']==2
    w=TrainingWindow();qtbot.addWidget(w);w.show();w.tabs.setCurrentIndex(1)
    w.apply_checkpoint.setText(str(checkpoint));w.apply_source.setText(ds['pairs'][0]['original']);w.apply_output.setText(str(tmp_path/'predictions'))
    w.apply_device.setCurrentIndex(w.apply_device.findData('cpu'));assert w.apply_run.isEnabled()
    w.load_classifier();qtbot.waitUntil(lambda:w.process is None,timeout=30000)
    assert w.apply_table.rowCount()==2 and 'Classifier loaded' in w.apply_status.text()
    w.apply_classifier();qtbot.waitUntil(lambda:w.process is None,timeout=60000)
    assert w.prediction_mask.is_file() and 'Classification completed' in w.apply_status.text()
    assert not w.original_viewer.image.pixmap().isNull() and not w.classified_viewer.image.pixmap().isNull()
    assert w.save_prediction.isEnabled()
    w.apply_directory.setText(str(tmp_path/'originals'));assert not w.apply_run.isEnabled()
    w.apply_source.clear();assert w.apply_run.isEnabled()
    w.apply_classifier();qtbot.waitUntil(lambda:w.process is None,timeout=60000)
    assert w.batch_output.is_dir() and 'Batch classification completed' in w.apply_status.text()
    manifest=json.loads((w.batch_output/'batch.json').read_text());assert len(manifest['images'])==3
    assert all(Path(item['output'],'prediction.tif').is_file() for item in manifest['images'])
    assert w.prediction_source==manifest['images'][-1]['source'] and w.save_prediction.isEnabled();w.close()


@pytest.mark.parametrize('device',['mps','cuda'])
def test_available_gpu_unet_step(device):
    import torch
    if device not in devices():pytest.skip(device+' unavailable')
    net=UNet(1,2,4,2).to(device);x=torch.zeros((1,1,16,16),device=device);target=torch.zeros((1,16,16),dtype=torch.long,device=device)
    loss=torch.nn.functional.cross_entropy(net(x),target);loss.backward();assert torch.isfinite(loss)


def test_native_failed_worker_recovers_and_changed_folders_block(qtbot,tmp_path,monkeypatch):
    from unet_trainer.window import TrainingWindow
    w=TrainingWindow();qtbot.addWidget(w);w.show();ds=dataset(tmp_path);w.populate(ds)
    w.paths['originals'].setText(str(tmp_path/'different'));w.paths['masks'].setText(str(tmp_path/'masks'))
    with pytest.raises(ValueError,match='folders changed'):w.named_dataset()
    with monkeypatch.context() as m:
        m.setattr('unet_trainer.window.sys.executable','/nonexistent-unet-python')
        w.launch_job({'mode':'inspect','originals':'','masks':''})
        qtbot.waitUntil(lambda:w.process is None,timeout=5000)
    assert w.config_file is None and w.controls.isEnabled() and not w.cancel.isEnabled()
    w.close()


def test_lzw_tiff_native_samples_palette_and_dataset(tmp_path):
    from unet_trainer.image_io import read_image,read_mask
    a=tmp_path/'originals';b=tmp_path/'masks';a.mkdir();b.mkdir()
    original=np.arange(120,dtype=np.uint16).reshape(10,12)*500
    mask=np.full((10,12),4,np.uint8);mask[2:8,2:10]=0;mask[4:6,4:6]=4
    palette=np.zeros((3,256),np.uint16);palette[:,0]=[65280,0,0];palette[:,4]=[21504,57856,65280]
    tifffile.imwrite(a/'image016.tif',original,compression='lzw')
    tifffile.imwrite(b/'Annotated016.tif',mask,compression='lzw',photometric='palette',colormap=palette)
    np.testing.assert_array_equal(read_image(a/'image016.tif'),original)
    np.testing.assert_array_equal(read_mask(b/'Annotated016.tif',mask.shape,{'#ff0000':0,'#54e2ff':4}),mask)
    ds=inspect_dataset(a,b);assert [c['id'] for c in ds['classes']]==[0,4]


def test_failed_inspect_records_actual_reason(qtbot,tmp_path):
    from unet_trainer.window import TrainingWindow
    w=TrainingWindow();qtbot.addWidget(w);w.show()
    w.launch_job({'mode':'inspect','originals':str(tmp_path/'missing'),'masks':str(tmp_path/'also-missing')})
    qtbot.waitUntil(lambda:w.process is None,timeout=30000)
    assert 'FAILED: Folder not found:' in w.log.toPlainText()
    assert 'Folder not found:' in w.diagnostic_path.read_text()
    assert w.diagnostic_path.with_suffix('.json').is_file()
    w.close()
