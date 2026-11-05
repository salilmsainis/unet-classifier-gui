"""Subprocess entry point: no Qt, explicit JSON events."""
import json,sys,traceback
from .core import checkpoint_metadata,inspect_dataset,train,predict,predict_directory

def emit(event):print(json.dumps(event),flush=True)
def main():
    config=json.loads(open(sys.argv[1]).read());mode=config['mode']
    if mode=='inspect':emit({'type':'dataset','dataset':inspect_dataset(config['originals'],config['masks'])})
    elif mode=='classifier':emit({'type':'classifier','metadata':checkpoint_metadata(config['checkpoint']),'checkpoint':config['checkpoint']})
    elif mode=='train':train(config,emit)
    elif mode=='predict':emit({'type':'prediction','output':str(predict(config['checkpoint'],config['source'],config['device'],config['output']))})
    elif mode=='predict_directory':
        output=predict_directory(config['checkpoint'],config['source_directory'],config['device'],config['output'],emit)
        manifest=json.loads((output/'batch.json').read_text());emit({'type':'batch_done','output':str(output),'count':len(manifest['images'])})
    else:raise ValueError('Unknown operation.')
if __name__=='__main__':
    try:main()
    except Exception as e:traceback.print_exc(file=sys.stderr);emit({'type':'error','message':str(e)});sys.exit(1)
