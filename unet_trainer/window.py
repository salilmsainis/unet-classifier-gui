"""Standalone native training and classifier-application dashboard."""
from pathlib import Path
import datetime,json,shutil,sys,tempfile
import numpy as np
from PyQt6.QtCore import Qt,QProcess,QTimer
from PyQt6.QtGui import QColor,QImage,QPainter,QPen,QPixmap
from PyQt6.QtWidgets import (QComboBox,QDoubleSpinBox,QFileDialog,QFormLayout,QGridLayout,QGroupBox,QHeaderView,QHBoxLayout,QLabel,QLineEdit,QMainWindow,QMessageBox,QPlainTextEdit,QProgressBar,QPushButton,QScrollArea,QSpinBox,QSplitter,QTabWidget,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget)
from .core import devices,normalize
from .image_io import load_labels,read_image


THEME='''QWidget{background:#20252d;color:#e8edf5;font-size:13px} QGroupBox{border:1px solid #414b5b;border-radius:8px;margin-top:14px;padding:12px} QGroupBox::title{subcontrol-origin:margin;left:12px;color:#a8c3e2} QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit,QTableWidget{background:#171c23;border:1px solid #414b5b;border-radius:4px;padding:5px} QPushButton{background:#344657;border:1px solid #526377;border-radius:5px;padding:8px} QPushButton:hover{background:#405b72} QPushButton:disabled{color:#74818d;background:#29313b} QHeaderView::section{background:#303b48;padding:5px} QProgressBar{border:1px solid #435366;border-radius:5px;text-align:center} QProgressBar::chunk{background:#24a99f} QTabWidget::pane{border:1px solid #414b5b;border-radius:7px;top:-1px} QTabBar::tab{background:#2a313b;border:1px solid #414b5b;border-bottom:none;border-top-left-radius:6px;border-top-right-radius:6px;padding:11px 24px;margin-right:3px;color:#aeb9c7} QTabBar::tab:selected{background:#344657;color:#f5f8fc;border-bottom:2px solid #6ee3d5} QScrollArea{border:1px solid #303946;border-radius:5px;background:#15191f}'''


def rgb_image(array):
    """Convert an original image to an 8-bit display copy without changing it."""
    a=np.moveaxis(normalize(array),0,-1)
    if a.shape[-1]==1:a=np.repeat(a,3,axis=-1)
    return np.ascontiguousarray((a*255).clip(0,255).astype(np.uint8))


def pixmap(array):
    a=np.ascontiguousarray(array)
    image=QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class ImageViewer(QGroupBox):
    """Small zoomable, pannable image viewer used by the Apply tab."""
    def __init__(self,title):
        super().__init__(title);self.original=None;self.zoom=1.;self.fitted=True
        layout=QVBoxLayout(self);tools=QHBoxLayout();self.info=QLabel('No image loaded');tools.addWidget(self.info);tools.addStretch()
        for text,callback in [('Fit',self.fit),('1:1',self.actual),('−',lambda:self.step(.8)),('+',lambda:self.step(1.25))]:
            button=QPushButton(text);button.setMaximumWidth(56);button.clicked.connect(callback);tools.addWidget(button)
        layout.addLayout(tools);self.scroll=QScrollArea();self.scroll.setWidgetResizable(False)
        self.image=QLabel('Choose an image or image directory, then apply the classifier.');self.image.setAlignment(Qt.AlignmentFlag.AlignCenter);self.image.setWordWrap(True);self.image.setMinimumSize(320,300);self.scroll.setWidget(self.image);layout.addWidget(self.scroll,1)
    def set_array(self,array,description):
        self.original=pixmap(array);self.info.setText(description);self.fitted=True;QTimer.singleShot(0,self.fit)
    def render(self):
        if not self.original:return
        shown=self.original.scaled(max(1,int(self.original.width()*self.zoom)),max(1,int(self.original.height()*self.zoom)),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        self.image.setPixmap(shown);self.image.resize(shown.size())
    def fit(self):
        if not self.original:return
        available=self.scroll.viewport().size();source=self.original.size()
        self.zoom=min(1.,max(.02,min((available.width()-8)/source.width(),(available.height()-8)/source.height())))
        self.fitted=True;self.render()
    def actual(self):
        self.zoom=1.;self.fitted=False;self.render()
    def step(self,factor):
        if not self.original:return
        self.zoom=max(.02,min(16.,self.zoom*factor));self.fitted=False;self.render()
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.fitted:QTimer.singleShot(0,self.fit)


class Plot(QWidget):
    def __init__(self,title,fixed=False):
        super().__init__();self.title=title;self.fixed=fixed;self.series=[];self.setMinimumSize(240,165)
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.fillRect(self.rect(),QColor('#20252d'));p.setPen(QColor('#c8d6e5'));p.drawText(10,20,self.title)
        left,top,right,bottom=45,40,self.width()-15,self.height()-40
        p.setPen(QColor('#526070'));p.drawLine(left,top,left,bottom);p.drawLine(left,bottom,right,bottom)
        values=[v for _,_,ys in self.series for v in ys if v is not None];maximum=1. if self.fixed else max([1.]+values)*1.05
        count=max([2]+[len(ys) for _,_,ys in self.series])
        for tick in (0,.5,1):p.drawText(3,int(bottom-(bottom-top)*tick),f'{maximum*tick:.2f}')
        p.drawText(right-55,bottom+17,'epoch');p.drawText(left,bottom+17,'1');p.drawText(right-12,bottom+17,str(count))
        for idx,(name,color,ys) in enumerate(self.series):
            p.setPen(QPen(QColor(color),2));last=None
            for i,y in enumerate(ys):
                if y is None:last=None;continue
                point=(int(left+(right-left)*i/(count-1)),int(bottom-(bottom-top)*y/maximum))
                if last:p.drawLine(*last,*point)
                p.drawEllipse(point[0]-2,point[1]-2,4,4);last=point
            p.drawText(left+(idx%3)*120,self.height()-5-15*(idx//3),name[:18])


class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('U-Net classifier');self.resize(1440,960);self.setStyleSheet(THEME)
        self.dataset=None;self.history=[];self.process=None;self.config_file=None;self.buffer='';self.mode=None;self.job_classes=[];self.run_output=None;self.had_error=False
        self.apply_classes=[];self.loaded_checkpoint=None;self.prediction_source=None;self.prediction_mask=None;self.batch_output=None
        root=QWidget();self.setCentralWidget(root);layout=QVBoxLayout(root);self.tabs=QTabWidget();layout.addWidget(self.tabs)
        self.tabs.addTab(self.build_training_page(),'Train classifier');self.tabs.addTab(self.build_apply_page(),'Apply classifier')
        self.checkpoint.textChanged.connect(self.sync_checkpoint_from_training);self.apply_checkpoint.textChanged.connect(self.sync_checkpoint_from_apply)
        self.device.currentIndexChanged.connect(lambda:self.sync_device(self.device,self.apply_device));self.apply_device.currentIndexChanged.connect(lambda:self.sync_device(self.apply_device,self.device))
        self.update_enabled()

    def build_training_page(self):
        page=QWidget();layout=QVBoxLayout(page);split=QSplitter();layout.addWidget(split)
        controls=QWidget();left=QVBoxLayout(controls);self.controls=controls
        data=QGroupBox('Data • numbered 2D image / mask pairs');dl=QVBoxLayout(data);left.addWidget(data);self.paths={}
        for key,title in [('originals','Original images folder'),('masks','Annotated masks folder'),('output','Output folder')]:
            dl.addWidget(QLabel(title));row=QHBoxLayout();field=QLineEdit()
            if key=='output':field.setPlaceholderText('Choose where training results will be saved')
            self.paths[key]=field;row.addWidget(field);button=QPushButton('Browse…');button.clicked.connect(lambda _,k=key:self.browse(k));row.addWidget(button);dl.addLayout(row)
        self.load=self.button('Load data / detect all classes',self.load_data,dl)
        self.summary=QLabel('Choose separate original and mask folders.\nThe last number in each filename identifies its pair.');self.summary.setWordWrap(True);dl.addWidget(self.summary)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['ID','Colour','Class name','Pixels']);self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents);self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.ResizeToContents);self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch);self.table.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeMode.ResizeToContents);self.table.setMinimumHeight(185);dl.addWidget(self.table)
        self.profile=self.button('Load class names from JSON…',self.load_profile,dl)
        model=QGroupBox('Model');form=QFormLayout(model);left.addWidget(model);self.device=self.device_combo();form.addRow('Processor',self.device);form.addRow('Architecture',QLabel('U-Net • trained from scratch'))
        self.base=QComboBox();self.base.addItems(['4','8','16','32','64']);self.base.setCurrentText('16');form.addRow('Base channels',self.base);self.settings={};self.add_spin(form,'depth','Depth',3,1,4)
        group=QGroupBox('Training');form=QFormLayout(group);left.addWidget(group)
        for key,title,value,lo,hi in [('epochs','Epochs',60,1,10000),('batch','Batch size',4,1,64),('patch','Patch size',128,16,1024),('patches','Patches / epoch',100,1,100000),('validation_percent','Validation images (%)',20,1,90),('seed','Seed',42,0,2147483647),('patience','Early stop after (0 = off)',20,0,1000)]:self.add_spin(form,key,title,value,lo,hi)
        for key,title,value in [('lr','Learning rate',.0003),('weight_decay','Weight decay',.0001)]:
            spin=QDoubleSpinBox();spin.setDecimals(7);spin.setRange(0.0000001 if key=='lr' else 0,1);spin.setValue(value);self.settings[key]=spin;form.addRow(title,spin)
        self.loss=QComboBox();self.loss.addItems(['Cross-entropy + Dice','Cross-entropy']);form.addRow('Loss',self.loss)
        hint=QLabel('Validation uses whole held-out images. For adjacent tomography slices, use a separate specimen for a convincing final evaluation. Source samples are never overwritten.');hint.setWordWrap(True);left.addWidget(hint)
        self.start=self.button('Train U-Net',self.start_training,left);reuse=QGroupBox('Saved classifier');rl=QVBoxLayout(reuse);left.addWidget(reuse)
        self.checkpoint=QLineEdit();self.checkpoint.setPlaceholderText('best.unet.pt');rl.addWidget(self.checkpoint);self.button('Open classifier…',self.choose_checkpoint,rl);self.button('Use in Apply classifier tab',self.use_classifier,rl)
        left.addStretch();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(controls);scroll.setMinimumWidth(410);split.addWidget(scroll)
        main=QWidget();right=QVBoxLayout(main);split.addWidget(main);split.setSizes([430,1000]);stats=QHBoxLayout();self.stats={}
        for key,title in [('epoch','EPOCH'),('best','BEST mIoU'),('mean_dice','MEAN DICE'),('val_loss','VAL LOSS'),('seconds','TIME / EPOCH'),('remaining','REMAINING')]:
            box=QGroupBox(title);bl=QVBoxLayout(box);label=QLabel('—');label.setStyleSheet('font-size:22px;color:#6ee3d5');bl.addWidget(label);stats.addWidget(box);self.stats[key]=label
        right.addLayout(stats);self.progress=QProgressBar();right.addWidget(self.progress);plots=QGridLayout();self.plots=[Plot('Loss'),Plot('Validation mIoU & mean Dice',True),Plot('IoU per class • validation',True),Plot('Dice per class • validation',True)]
        for i,plot in enumerate(self.plots):plots.addWidget(plot,i//2,i%2)
        right.addLayout(plots,2);previews=QHBoxLayout();self.previews=[]
        for title in ('VALIDATION IMAGE','GROUND TRUTH','PREDICTION'):
            box=QGroupBox(title);bl=QVBoxLayout(box);label=QLabel('No completed epoch yet');label.setAlignment(Qt.AlignmentFlag.AlignCenter);label.setMinimumSize(160,150);bl.addWidget(label);previews.addWidget(box);self.previews.append(label)
        right.addLayout(previews,1);self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(1000);self.log.setMaximumHeight(150);right.addWidget(self.log);self.cancel=self.button('Cancel current job',self.cancel_job,right)
        return page

    def build_apply_page(self):
        page=QWidget();layout=QVBoxLayout(page);split=QSplitter();layout.addWidget(split);controls=QWidget();left=QVBoxLayout(controls);self.apply_controls=controls
        classifier=QGroupBox('Classifier');form=QFormLayout(classifier);left.addWidget(classifier);row=QHBoxLayout();self.apply_checkpoint=QLineEdit();self.apply_checkpoint.setPlaceholderText('best.unet.pt');row.addWidget(self.apply_checkpoint);button=QPushButton('Browse…');button.clicked.connect(lambda:self.guard(self.choose_apply_checkpoint));row.addWidget(button);form.addRow('Classifier file',row)
        self.apply_device=self.device_combo();form.addRow('Processor',self.apply_device);self.load_classifier_button=QPushButton('Load classifier');self.load_classifier_button.clicked.connect(lambda:self.guard(self.load_classifier));form.addRow(self.load_classifier_button)
        inputs=QGroupBox('Input image(s) and output');form=QFormLayout(inputs);left.addWidget(inputs);row=QHBoxLayout();self.apply_source=QLineEdit();self.apply_source.setPlaceholderText('2D TIFF, PNG or JPEG');row.addWidget(self.apply_source);button=QPushButton('Browse…');button.clicked.connect(lambda:self.guard(self.choose_apply_source));row.addWidget(button);form.addRow('Image file (one)',row)
        row=QHBoxLayout();self.apply_directory=QLineEdit();self.apply_directory.setPlaceholderText('Folder of 2D TIFF, PNG or JPEG images');row.addWidget(self.apply_directory);button=QPushButton('Browse…');button.clicked.connect(lambda:self.guard(self.choose_apply_directory));row.addWidget(button);form.addRow('Image directory (batch)',row)
        row=QHBoxLayout();self.apply_output=QLineEdit();self.apply_output.setPlaceholderText('Choose where classification results will be saved');row.addWidget(self.apply_output);button=QPushButton('Browse…');button.clicked.connect(lambda:self.guard(self.choose_apply_output));row.addWidget(button);form.addRow('Output folder',row)
        self.apply_run=QPushButton('Apply classifier');self.apply_run.clicked.connect(lambda:self.guard(self.apply_classifier));form.addRow(self.apply_run)
        classes=QGroupBox('Classifier classes');cl=QVBoxLayout(classes);left.addWidget(classes);self.apply_table=QTableWidget(0,3);self.apply_table.setHorizontalHeaderLabels(['ID','Colour','Class name']);self.apply_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents);self.apply_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.ResizeToContents);self.apply_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch);self.apply_table.setMinimumHeight(190);cl.addWidget(self.apply_table)
        note=QLabel('The classified image uses the colours stored with the model. The saved TIFF retains numeric class IDs.');note.setWordWrap(True);left.addWidget(note);left.addStretch();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(controls);scroll.setMinimumWidth(420);split.addWidget(scroll)
        workspace=QWidget();right=QVBoxLayout(workspace);headline=QHBoxLayout();title=QLabel('Classification preview');title.setStyleSheet('font-size:18px;color:#f5f8fc');headline.addWidget(title);headline.addStretch();self.save_prediction=QPushButton('Save classified image…');self.save_prediction.clicked.connect(lambda:self.guard(self.save_classified));headline.addWidget(self.save_prediction);right.addLayout(headline)
        viewers=QHBoxLayout();self.original_viewer=ImageViewer('ORIGINAL IMAGE');self.classified_viewer=ImageViewer('CLASSIFIED IMAGE');viewers.addWidget(self.original_viewer,1);viewers.addWidget(self.classified_viewer,1);right.addLayout(viewers,1)
        self.apply_progress=QProgressBar();right.addWidget(self.apply_progress);status=QHBoxLayout();self.apply_status=QLabel('Load a classifier and choose an image or image directory.');self.apply_status.setWordWrap(True);status.addWidget(self.apply_status,1);self.apply_cancel=QPushButton('Cancel current job');self.apply_cancel.clicked.connect(self.cancel_job);status.addWidget(self.apply_cancel);right.addLayout(status);split.addWidget(workspace);split.setSizes([430,1000])
        self.apply_checkpoint.textChanged.connect(self.mark_classifier_changed);self.apply_source.textChanged.connect(self.update_enabled);self.apply_directory.textChanged.connect(self.update_enabled)
        return page

    def device_combo(self):
        combo=QComboBox();combo.addItem('Choose processor…',None);available=set(devices())
        for name,label in [('cpu','CPU'),('cuda','NVIDIA CUDA'),('mps','Apple Silicon MPS')]:
            index=combo.count();combo.addItem(label if name in available else label+' (unavailable)',name)
            if name not in available:
                item=combo.model().item(index)
                if item:item.setEnabled(False)
                combo.setItemData(index,'Install a compatible PyTorch build and driver to enable this processor.',Qt.ItemDataRole.ToolTipRole)
        combo.currentIndexChanged.connect(self.update_enabled);return combo
    def button(self,text,callback,layout):
        button=QPushButton(text);button.clicked.connect(lambda:self.guard(callback));layout.addWidget(button);return button
    def guard(self,callback):
        try:callback()
        except Exception as error:self.log.appendPlainText(str(error));QMessageBox.warning(self,'U-Net classifier',str(error))
    def add_spin(self,form,key,title,value,low,high):
        spin=QSpinBox();spin.setRange(low,high);spin.setValue(value);self.settings[key]=spin;form.addRow(title,spin)
    def sync_checkpoint_from_training(self,text):
        if self.apply_checkpoint.text()!=text:self.apply_checkpoint.setText(text)
    def sync_checkpoint_from_apply(self,text):
        if self.checkpoint.text()!=text:self.checkpoint.setText(text)
    def sync_device(self,source,target):
        index=target.findData(source.currentData())
        if index>=0 and target.currentIndex()!=index:target.setCurrentIndex(index)
    def mark_classifier_changed(self):
        if self.loaded_checkpoint!=self.apply_checkpoint.text():self.loaded_checkpoint=None
        self.update_enabled()
    def browse(self,key):
        path=QFileDialog.getExistingDirectory(self,'Choose '+key,self.paths[key].text())
        if path:self.paths[key].setText(path)
    def load_data(self):
        self.dataset=None;self.table.setRowCount(0);self.update_enabled();self.launch_job({'mode':'inspect','originals':self.paths['originals'].text(),'masks':self.paths['masks'].text()})
    def populate(self,dataset):
        self.dataset=dataset;self.table.setRowCount(len(dataset['classes']))
        for row,item in enumerate(dataset['classes']):
            for column,value in enumerate((item['id'],item['color'],item['name'],item['count'])):
                cell=QTableWidgetItem(str(value))
                if column!=2:cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column==1:cell.setForeground(QColor(item['color']))
                self.table.setItem(row,column,cell)
        self.table.resizeColumnsToContents();self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch);self.summary.setText(f"{len(dataset['pairs'])} matched pairs • {len(dataset['classes'])} classes. Edit names in the table. IDs stay distinct even when names match.");self.update_enabled()
    def populate_apply_classes(self,classes):
        self.apply_classes=classes;self.apply_table.setRowCount(len(classes))
        for row,item in enumerate(classes):
            for column,value in enumerate((item['id'],item['color'],item['name'])):
                cell=QTableWidgetItem(str(value));cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column==1:cell.setForeground(QColor(item['color']));cell.setBackground(QColor(item['color']).darker(250))
                self.apply_table.setItem(row,column,cell)
        self.apply_table.resizeColumnsToContents();self.apply_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch)
    def load_profile(self):
        if not self.dataset:raise ValueError('Load data first.')
        path,_=QFileDialog.getOpenFileName(self,'Choose class names','','JSON (*.json)')
        if not path:return
        definitions,mapping=load_labels(path)
        for row,item in enumerate(self.dataset['classes']):
            key=mapping.get(item['color'],item['id']);entry=definitions.get(key)
            if entry:self.table.item(row,2).setText(entry['name'])
    def named_dataset(self):
        if not self.dataset:raise ValueError('Load data first.')
        if self.dataset['pairs']:
            pair=self.dataset['pairs'][0]
            if Path(self.paths['originals'].text()).expanduser().resolve()!=Path(pair['original']).parent.resolve() or Path(self.paths['masks'].text()).expanduser().resolve()!=Path(pair['mask']).parent.resolve():raise ValueError('Input folders changed. Click Load data again before training.')
        dataset=json.loads(json.dumps(self.dataset))
        for row,item in enumerate(dataset['classes']):
            item['name']=self.table.item(row,2).text().strip()
            if not item['name']:raise ValueError('Every class needs a name.')
        return dataset
    def destination(self,prefix,base=None):
        value=self.paths['output'].text() if base is None else base
        if not value.strip():raise ValueError('Choose an output folder.')
        return str(Path(value).expanduser().resolve()/(prefix+'-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    def start_training(self):
        dataset=self.named_dataset()
        if len(dataset['pairs'])<2:raise ValueError('At least two annotated images are required for separate training and validation images.')
        config={key:spin.value() for key,spin in self.settings.items()};config['validation']=config.pop('validation_percent')/100;config.update(mode='train',dataset=dataset,device=self.device.currentData(),base=int(self.base.currentText()),loss=self.loss.currentText(),output=self.destination('run'))
        if config['device'] is None:raise ValueError('Choose CPU, NVIDIA CUDA or Apple MPS.')
        if config['patch']<2**config['depth']:raise ValueError('Patch size is too small for the chosen depth.')
        self.job_classes=dataset['classes'];self.run_output=config['output'];self.history=[]
        for label in self.stats.values():label.setText('—')
        for plot in self.plots:plot.series=[];plot.update()
        for label in self.previews:label.clear();label.setText('Waiting for validation')
        self.launch_job(config)
    def choose_checkpoint(self):
        path,_=QFileDialog.getOpenFileName(self,'Open U-Net classifier','','U-Net classifier (*.pt)')
        if path:self.checkpoint.setText(path)
    def use_classifier(self):
        if not self.checkpoint.text().strip():raise ValueError('Open or train a classifier first.')
        self.tabs.setCurrentIndex(1);self.load_classifier()
    def choose_apply_checkpoint(self):
        path,_=QFileDialog.getOpenFileName(self,'Open U-Net classifier','','U-Net classifier (*.pt)')
        if path:self.apply_checkpoint.setText(path)
    def choose_apply_source(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose a new 2D image','','Images (*.tif *.tiff *.png *.jpg *.jpeg)')
        if path:self.apply_directory.clear();self.apply_source.setText(path)
    def choose_apply_directory(self):
        path=QFileDialog.getExistingDirectory(self,'Choose a directory of 2D images',self.apply_directory.text())
        if path:self.apply_source.clear();self.apply_directory.setText(path)
    def choose_apply_output(self):
        path=QFileDialog.getExistingDirectory(self,'Choose output folder',self.apply_output.text())
        if path:self.apply_output.setText(path)
    def load_classifier(self):
        path=self.apply_checkpoint.text().strip()
        if not path:raise ValueError('Choose a saved classifier first.')
        self.apply_status.setText('Loading classifier metadata…');self.launch_job({'mode':'classifier','checkpoint':path})
    def apply_classifier(self):
        checkpoint=self.apply_checkpoint.text().strip();source=self.apply_source.text().strip();source_directory=self.apply_directory.text().strip();device=self.apply_device.currentData()
        if not checkpoint:raise ValueError('Choose a saved classifier first.')
        if not source and not source_directory:raise ValueError('Choose one 2D image or an input image directory first.')
        if source and source_directory:raise ValueError('Choose either one image or an image directory, not both.')
        if device is None:raise ValueError('Choose the processor.')
        self.prediction_mask=None;self.batch_output=None
        if source:
            self.prediction_source=source;self.apply_status.setText('Applying classifier…');config={'mode':'predict','checkpoint':checkpoint,'source':source,'device':device,'output':self.destination('prediction',self.apply_output.text())}
        else:
            self.prediction_source=None;self.apply_status.setText('Applying classifier to image directory…');config={'mode':'predict_directory','checkpoint':checkpoint,'source_directory':source_directory,'device':device,'output':self.destination('prediction-batch',self.apply_output.text())}
        self.launch_job(config)
    def save_classified(self):
        if not self.prediction_mask or not self.prediction_mask.is_file():raise ValueError('Apply a classifier first.')
        suggested=Path(self.prediction_source).stem+'-classified.tif';path,_=QFileDialog.getSaveFileName(self,'Save classified image',str(Path(self.apply_output.text()).expanduser()/suggested),'TIFF label image (*.tif *.tiff)')
        if path:
            if Path(path).suffix.lower() not in ('.tif','.tiff'):path+='.tif'
            shutil.copy2(self.prediction_mask,path);self.apply_status.setText('Classified TIFF saved to '+path)
    def launch_job(self,config):
        if self.process is not None:raise ValueError('Wait for the running job to finish.')
        self.mode=config['mode'];self.had_error=False;self.buffer='';self.last_error='';self.stderr_tail='';self.cancelled=False
        diagnostics=Path(__file__).resolve().parents[1]/'Diagnostics';diagnostics.mkdir(exist_ok=True);stem=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-'+self.mode;self.diagnostic_path=diagnostics/(stem+'.log')
        (diagnostics/(stem+'.json')).write_text(json.dumps(config,indent=2));self.diagnostic_path.write_text('Starting '+self.mode+'\n')
        with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as stream:json.dump(config,stream);self.config_file=Path(stream.name)
        self.process=QProcess(self);self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]));self.process.setProgram(sys.executable);self.process.setArguments(['-m','unet_trainer.worker',str(self.config_file)]);self.process.readyReadStandardOutput.connect(self.read_events);self.process.readyReadStandardError.connect(self.read_errors);self.process.finished.connect(self.finished);self.process.errorOccurred.connect(self.process_error)
        for progress in (self.progress,self.apply_progress):progress.setRange(0,0)
        self.log.appendPlainText('Starting '+self.mode+'…');self.process.start();self.update_enabled()
    def process_error(self,error):
        if not self.process:return
        self.last_error=self.process.errorString();self.record_log('Process error: '+self.last_error)
        if error==QProcess.ProcessError.FailedToStart:self.had_error=True;self.finished(-1,QProcess.ExitStatus.NormalExit)
    def record_log(self,text):
        self.log.appendPlainText(text)
        try:
            with self.diagnostic_path.open('a') as stream:stream.write(text+'\n')
        except OSError:pass
    def read_errors(self):
        if self.process:
            text=bytes(self.process.readAllStandardError()).decode(errors='replace')
            if text:self.stderr_tail=(self.stderr_tail+text)[-16000:];self.record_log(text)
    def read_events(self):
        if not self.process:return
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try:
                with self.diagnostic_path.open('a') as stream:stream.write(line+'\n')
            except OSError:pass
            try:event=json.loads(line)
            except ValueError:self.log.appendPlainText(line);continue
            kind=event.get('type')
            if kind=='dataset':self.populate(event['dataset'])
            elif kind=='classifier':
                self.loaded_checkpoint=event['checkpoint'];self.populate_apply_classes(event['metadata']['classes']);architecture=event['metadata']['architecture'];self.apply_status.setText(f"Classifier loaded • {architecture['channels']} input channel(s) • {architecture['classes']} classes • patch {event['metadata']['patch']}")
            elif kind=='epoch':self.epoch(event)
            elif kind=='done':self.checkpoint.setText(event['checkpoint']);self.log.appendPlainText('Completed. Best classifier: '+event['checkpoint'])
            elif kind=='prediction':self.show_application_result(Path(event['output']))
            elif kind=='batch_prediction':
                self.prediction_source=event['source'];self.show_application_result(Path(event['output']),False);self.apply_progress.setRange(0,event['total']);self.apply_progress.setValue(event['index']);self.apply_status.setText(f"Classified {event['index']} of {event['total']} • {Path(event['source']).name}")
            elif kind=='batch_done':
                self.batch_output=Path(event['output']);self.apply_status.setText(f"Batch classification completed • {event['count']} images saved in {event['output']}");self.log.appendPlainText('Batch prediction saved: '+event['output'])
            elif kind=='error':self.had_error=True;self.last_error=event['message'];self.log.appendPlainText('ERROR: '+self.last_error)
    def epoch(self,row):
        self.history.append(row)
        for key,label in self.stats.items():label.setText(str(row[key]) if key=='epoch' else (f'{row[key]:.1f}s' if key in ('seconds','remaining') else f'{row[key]:.4f}'))
        self.progress.setRange(0,self.settings['epochs'].value());self.progress.setValue(row['epoch']);self.plots[0].series=[('train','#6ca9ff',[item['train_loss'] for item in self.history]),('validation','#ffb668',[item['val_loss'] for item in self.history])];self.plots[1].series=[('mIoU','#6ca9ff',[item['miou'] for item in self.history]),('mean Dice','#ffb668',[item['mean_dice'] for item in self.history])]
        for plot,key in zip(self.plots[2:],('iou','dice')):plot.series=[(str(item['id'])+' '+item['name'],item['color'],[entry[key][i] for entry in self.history]) for i,item in enumerate(self.job_classes)]
        for plot in self.plots:plot.update()
        self.checkpoint.setText(str(Path(row['output'])/'best.unet.pt'));self.show_preview(Path(row['output'])/'preview.npz');self.log.appendPlainText(f"Epoch {row['epoch']}: validation mIoU {row['miou']:.4f}, Dice {row['mean_dice']:.4f}")
    def show_preview(self,path):
        try:
            with np.load(path) as data:
                image=(data['image']*255).clip(0,255).astype(np.uint8)
                if image.shape[-1]==1:image=np.repeat(image,3,axis=-1)
                palette=np.asarray([[int(item['color'][i:i+2],16) for i in (1,3,5)] for item in self.job_classes],np.uint8);arrays=[image,palette[data['truth']],palette[data['prediction']]]
            for label,array in zip(self.previews,arrays):label.setPixmap(pixmap(array).scaled(label.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        except (OSError,ValueError) as error:self.log.appendPlainText('Preview not available: '+str(error))
    def show_application_result(self,output,completed=True):
        source=Path(self.prediction_source);classes=json.loads((output/'classes.json').read_text());mask=read_image(output/'prediction.tif')
        if mask.ndim!=2:raise ValueError('Prediction output is not a 2D label image.')
        color=np.full((*mask.shape,3),[255,0,255],np.uint8)
        for item in classes:
            rgb=np.asarray([int(item['color'][i:i+2],16) for i in (1,3,5)],np.uint8);color[mask==item['id']]=rgb
        original=read_image(source);self.populate_apply_classes(classes);self.original_viewer.set_array(rgb_image(original),f'{source.name} • {original.shape[1]} × {original.shape[0]}');self.classified_viewer.set_array(color,f'{len(classes)} classes • {mask.shape[1]} × {mask.shape[0]}')
        self.prediction_mask=output/'prediction.tif'
        if completed:self.apply_status.setText('Classification completed • numeric label TIFF and probabilities saved in '+str(output))
        self.log.appendPlainText('Prediction saved: '+str(output));self.tabs.setCurrentIndex(1);self.update_enabled()
    def finished(self,code,status):
        mode=self.mode;self.read_events();self.read_errors();self.record_log(f'Job exited with code {code}.')
        if code or self.had_error:
            reason=self.last_error or (self.stderr_tail.strip().splitlines()[-1] if self.stderr_tail.strip() else 'Worker stopped without an error message.');self.record_log('Cancelled by user.' if self.cancelled else 'FAILED: '+reason);self.record_log('Any earlier best checkpoint is retained; partial files are not final results.')
            if mode in ('classifier','predict','predict_directory'):self.apply_status.setText(('Cancelled.' if self.cancelled else 'Failed: '+reason))
        self.record_log('Diagnostic log: '+str(self.diagnostic_path))
        if self.config_file:self.config_file.unlink(missing_ok=True);self.config_file=None
        process=self.process;self.process=None;process.deleteLater()
        for progress in (self.progress,self.apply_progress):progress.setRange(0,100);progress.setValue(100 if code==0 and not self.had_error else 0)
        self.update_enabled()
    def update_enabled(self,*args):
        if not hasattr(self,'apply_controls'):return
        busy=self.process is not None;single=bool(self.apply_source.text().strip());batch=bool(self.apply_directory.text().strip());self.controls.setEnabled(not busy);self.apply_controls.setEnabled(not busy);self.cancel.setEnabled(busy);self.apply_cancel.setEnabled(busy);self.start.setEnabled(not busy and self.dataset is not None and self.device.currentData() is not None);self.load_classifier_button.setEnabled(not busy and bool(self.apply_checkpoint.text().strip()));self.apply_run.setEnabled(not busy and bool(self.apply_checkpoint.text().strip()) and single!=batch and self.apply_device.currentData() is not None);self.save_prediction.setEnabled(not busy and self.prediction_mask is not None and self.prediction_mask.is_file())
    def cancel_job(self):
        if self.process:
            self.cancelled=True;process=self.process;process.terminate();QTimer.singleShot(2000,lambda:process.kill() if self.process is process and process.state()!=QProcess.ProcessState.NotRunning else None);self.log.appendPlainText('Cancellation requested; retaining already completed best checkpoint.')
    def closeEvent(self,event):
        if self.process:QMessageBox.information(self,'Job running','Cancel the current job and wait for it to stop before closing.');event.ignore()
        else:event.accept()
