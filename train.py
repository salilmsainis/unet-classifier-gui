from pathlib import Path
import sys

def main():
    from unet_trainer.runtime import ensure_native_runtime
    ensure_native_runtime()
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--smoke-output');parser.add_argument('--smoke-tab',choices=('train','apply'),default='train');args=parser.parse_args()
    from unet_trainer.window import TrainingWindow
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    if args.check:print('U-Net trainer dependencies OK');return 0
    app=QApplication(sys.argv[:1]);w=TrainingWindow();w.show()
    if args.smoke_output:
        if args.smoke_tab=='apply':w.tabs.setCurrentIndex(1)
        def finish():
            ok=w.grab().save(args.smoke_output);w.close();app.exit(0 if ok else 1)
        QTimer.singleShot(1500,finish)
    return app.exec()
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc();print('See TRAINING MANUAL.txt for setup. No dependencies were installed.');raise SystemExit(1)
