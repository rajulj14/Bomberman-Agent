import sys, time, importlib, logging, numpy as np
sys.path.insert(0,'.')
logging.disable(logging.INFO)
from environment import BombeRLeWorld
class A:
    no_gui=True; scenario='classic'; seed=42; log_dir='logs'; match_name=None; save_replay=False
    save_stats=False; continue_without_training=True; silence_errors=False; turn_based=False
    update_interval=0.1; make_video=False; skip_frames=True; train=0; command_name='play'
name=sys.argv[1]; rounds=int(sys.argv[2])
mod=importlib.import_module(f'agent_code.{name}.callbacks')
orig=mod.act; times=[]
def timed(self,gs):
    t=time.perf_counter(); r=orig(self,gs); times.append((time.perf_counter()-t)*1000); return r
mod.act=timed
w=BombeRLeWorld(A(),[(name,False),('rule_based_agent',False),('coin_collector_agent',False),('peaceful_agent',False)])
for r in range(rounds):
    w.new_round()
    while w.running: w.do_step()
w.end()
t=np.array(times)
print(f"{name}: {len(t)} decisions, mean {t.mean():.2f} ms, median {np.median(t):.2f}, p99 {np.percentile(t,99):.2f}, max {t.max():.2f}")
print("first 100 steps: mean %.2f, max %.2f" % (t[:100].mean(), t[:100].max()))
