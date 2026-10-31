"""Partition independent selector trajectories across CPU processes."""
import argparse,json,time
import torch
import arcfx_selector_compare as r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-index',type=int,required=True)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    torch.set_num_threads(r.CFG['threads']);data=r.Data()
    manifest=json.loads((r.OUT/'manifest.json').read_text())
    assert manifest['config']==r.CFG
    for path,expected in manifest['source_data_hashes'].items():assert r.sha(r.ROOT/path)==expected,path
    ledger=r.OUT/f'worker_{args.worker_index}_ledger.jsonl'
    def event(**values):
        with ledger.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=time.time(),worker=args.worker_index,**values),default=str)+'\n')
    r.event=event
    jobs=[(backbone,mode,tau,seed) for backbone in r.CFG['backbones'] for seed in r.CFG['seeds'] for mode,tau in r.candidates()]
    for index,(backbone,mode,tau,seed) in enumerate(jobs):
        if index%args.workers!=args.worker_index:continue
        r.run_trajectory(data,backbone,mode,tau,seed)
    r.dump(r.OUT/f'worker_{args.worker_index}_complete.json',dict(complete=True,worker=args.worker_index))


if __name__=='__main__':main()
