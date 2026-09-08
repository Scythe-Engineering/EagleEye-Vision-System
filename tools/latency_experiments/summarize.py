"""Summarize complete, common-frame windows; exclude startup/transition samples."""
import csv,json,statistics,sys
from pathlib import Path
root=Path('/Users/dark/Custom-Apps/EagleEye-Java-Sim/evidence/latency-experiments-2026-09-08')
p=max(root.glob('lanes-*.csv'),key=lambda p:p.stat().st_mtime)
s=p.read_text();rows=list(csv.DictReader(s[:s.rfind('\n')+1].splitlines()))
scenario=int(sys.argv[1]); duration=int(sys.argv[2]) if len(sys.argv)>2 else 40
rows=[r for r in rows if r['scenario']==str(scenario)]
end=max(int(r['epoch_ms']) for r in rows)-1000; start=end-duration*1000
first=min(int(r['epoch_ms']) for r in rows)
if start<first+5000: print(f'WAIT: only {(end-first)/1000:.1f}s available; need {duration+5}s');sys.exit(2)
rows=[r for r in rows if start<=int(r['epoch_ms'])<end]
def stat(v):
 v=sorted(v)
 def q(p):
  x=(len(v)-1)*p; i=int(x); return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
 return dict(n=len(v),mean=statistics.mean(v),p50=q(.5),p95=q(.95),p99=q(.99),min=v[0],max=v[-1]) if v else dict(n=0)
lanes={}
for r in rows:
 k=r['lane']+'/'+r['kind']; lanes.setdefault(k,{})[int(r['seq'])]=r
# Compare identical sequence numbers to avoid attributing boundary/loss differences to latency.
comparison_keys=['robot20/joined','scheduled5/joined','thread5/joined','thread1/joined','event/joined','udp/frame']
common=set.intersection(*(set(lanes.get(k,{})) for k in comparison_keys))
result={'source':p.name,'scenario':scenario,'duration_s':duration,'start_ms':start,'end_ms':end,'common_frames':len(common),'lanes':{}}
for k,rr in lanes.items():
 selected=[r for seq,r in rr.items() if seq in common] if k in comparison_keys else list(rr.values())
 result['lanes'][k]={'all_unique_frames':len(rr),
  'capture_to_consume_ms':stat([(int(r['consume_us'])-int(r['capture_us']))/1000 for r in selected]),
  'ready_to_consume_ms':stat([(int(r['consume_us'])-int(r['ready_us']))/1000 for r in selected])}
base=[lanes['robot20/joined'][seq] for seq in common]
result['capture_to_ready_ms']=stat([(int(r['ready_us'])-int(r['capture_us']))/1000 for r in base])
(root/f'scenario-{scenario}.json').write_text(json.dumps(result,indent=2)+'\n')
with (root/f'scenario-{scenario}.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('scenario',scenario,'common',len(common),'Pi capture-to-ready',result['capture_to_ready_ms'])
for k in comparison_keys:
 v=result['lanes'].get(k,{});a=v.get('capture_to_consume_ms',{});b=v.get('ready_to_consume_ms',{})
 print(k,'total mean/p95',round(a.get('mean',0),2),round(a.get('p95',0),2),'post mean/p95',round(b.get('mean',0),2),round(b.get('p95',0),2),'frames',v.get('all_unique_frames'))
print('anomalies',{k:len(v) for k,v in lanes.items() if 'mismatch' in k or 'reordered' in k})
