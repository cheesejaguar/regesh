"""Standalone research figures from completed, frozen numerical artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--output',type=Path,default=Path('results/figures'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'regesh-extended-v1'})
    colors=['#28649b','#ba6b31']
    def save(fig,name):
        fig.tight_layout()
        svg=a.output/(name+'.svg')
        fig.savefig(svg,metadata={'Date':None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(a.output/(name+'.png'),dpi=150)
        plt.close(fig)
    profile=read(a.results/'memory-profile/summary.json');snapshot=read(a.results/'snapshot-profile/results.json')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,work,title in zip(axes[:2],['append','clocked'],['Append-only history','Observe + elapsed time']):
        for backend,label,color in [('snapshot','Full poststates',colors[1]),('python','Compact deltas',colors[0])]:
            values=[profile['measurements'][work][str(n)][backend]['database_bytes']/1e6 for n in profile['counts']]
            ax.plot(profile['counts'],values,'o-',label=label,color=color)
        ax.set(title=title,xlabel='Observed events',ylabel='Checkpointed SQLite MB');ax.legend(fontsize=8)
    rows=snapshot['rows'];axes[2].bar(['Old lookup','Latest hash'],[r['peak_rss_kib']/1024 for r in rows],color=[colors[1],colors[0]])
    axes[2].set(title='Snapshot process peak RSS',ylabel='MiB (250 events)')
    fig.suptitle('Memory representation and lookup gains; Rust has the same retained bytes',y=1.03)
    save(fig,'memory')
    summary=read(a.results/'software_real/summary.json');config=read(a.results/'software_real/config.json')
    rows=[json.loads(line) for line in (a.results/'software_real/results.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[2,1]});x=np.arange(len(config['tasks']))
    for j,condition in enumerate(config['conditions']):
        values=[100*np.mean([r['final_hidden_pass_rate'] for r in rows if r['condition']==condition and r['task']==t]) for t in config['tasks']]
        axes[0].bar(x+(j-.5)*.36,values,.36,label=condition,color=colors[j])
    axes[0].set(xticks=x,xticklabels=config['tasks'],ylabel='Hidden checks passed (%)',ylim=(0,100),title='Real model: six held-out software families');axes[0].legend(fontsize=8)
    d=summary['paired_final_pass_rate'];mean=100*d['hybrid_minus_ordinary'];lo,hi=np.array(d['ci95'])*100
    axes[1].axvline(0,color='#777',linestyle='--');axes[1].errorbar(mean,0,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color=colors[0])
    axes[1].set(xlabel='Hybrid minus ordinary (percentage points)',yticks=[],xlim=(-10,10),ylim=(-1,1),title='Paired mean and 95% interval')
    axes[1].text(0,-.55,'6 task clusters; 12 paired trials\nNo full-suite success in either condition',ha='center',fontsize=9)
    save(fig,'software')
    scores=read(a.results/'goemotions_baseline/metrics.json');fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,keys,title in [(axes[0],['micro_f1','macro_f1','exact_set_accuracy','mean_sample_jaccard'],'Recognition / exactness (higher is better)'),(axes[1],['hamming_loss','brier','micro_ece'],'Loss / calibration error (lower is better)')]:
        x=np.arange(len(keys))
        for j,(key,label) in enumerate([('reference','Learned reference'),('constant_prior','Constant prior')]):
            ax.bar(x+(j-.5)*.36,[scores[key][k] for k in keys],.36,label=label,color=colors[j])
        ax.set(xticks=x,xticklabels=[k.replace('_','\n') for k in keys],title=title);ax.legend(fontsize=8)
    fig.suptitle('Human-label recognition: gains and adverse tradeoffs (5,427 test comments)',y=1.03)
    save(fig,'recognition')
    qa=read(a.results/'multiaxis/psych_axes.json');fig,axes=plt.subplots(1,2,figsize=(11,4));variants=['base','permuted','persona'];x=np.arange(3)
    strict=[qa['per_variant'][v]['strict_primary']['accuracy'] for v in variants]
    choice=[qa['per_variant'][v]['exact_json_fence_choice']['accuracy_all_attempts'] for v in variants]
    axes[0].bar(x-.18,np.array(strict)*100,.36,label='Strict schema',color=colors[1]);axes[0].bar(x+.18,np.array(choice)*100,.36,label='Secondary fenced choice',color=colors[0])
    axes[0].set(xticks=x,xticklabels=variants,ylim=(0,100),ylabel='All-attempt keyed accuracy (%)',title='QA: 12 original questions and paired variants');axes[0].legend(fontsize=8)
    q=qa['secondary_diagnostics']['exact_json_fence_choice'];axes[1].bar(['Strict schema','Extractable choice'],[0,100*q['extractable_choice_coverage']],color=[colors[1],colors[0]])
    axes[1].set(ylim=(0,100),ylabel='Response coverage (%)',title='Format coverage limits interpretation')
    fig.suptitle('Protocol failure is distinct from emotional knowledge; no personality/experience inference',y=1.03)
    save(fig,'psychology')
    print('Saved four SVG/PNG figure pairs to',a.output)


if __name__=='__main__':main()
