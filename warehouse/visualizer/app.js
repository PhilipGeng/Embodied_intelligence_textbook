(() => {
  'use strict';
  const data = window.RL_DATA;
  const $ = id => document.getElementById(id);
  const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (x, n=3) => Number.isFinite(x) ? x.toFixed(n) : '—';
  const signed = x => Number.isFinite(x) ? `${x >= 0 ? '+' : ''}${x.toFixed(3)}` : '—';
  const percent = x => Number.isFinite(x) ? `${(100*x).toFixed(1)}%` : '—';
  const actions = ['↑ 上','→ 右','↓ 下','← 左'];
  const colors = ['#2563eb','#be5a16','#087f5b','#7c3aed','#c2413b','#067f97','#a24786','#64748b','#5b7307'];
  const initial = data.runs.findIndex(r => r.algorithm === 'ppo');
  const state = {run: initial < 0 ? 0 : initial, mode:'greedy', step:0, timer:null, bellman:0, tab:'replay', ppoManual:false};
  const curveEnabled = new Set(data.runs.filter(r => r.learning.length).map(r => r.id));
  const selectedRun = () => data.runs[state.run];
  const selectedEpisode = () => selectedRun()[state.mode] || selectedRun().greedy;
  const selectedFrame = () => selectedEpisode().frames[state.step];
  const position = p => `(${p[0]}, ${p[1]}, ${p[2] ? '载货' : '空载'})`;

  function drawMap(id, {current=null, previous=null, route=[], values=null, phase=0, world=null, predicted=null}) {
    const cell=68, ox=42, oy=26, W=560, H=400;
    const cx=c=>ox+(c+.5)*cell, cy=r=>oy+(r+.5)*cell;
    const list = world || (values ? values.slice(phase*35,phase*35+35) : []);
    const max = list.length ? Math.max(...list.map(Math.abs),1e-8) : 1;
    let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="仓库状态，行列从零开始"><defs><marker id="arrow-${id}" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5" fill="${phase?'#7c3aed':'#2563eb'}"/></marker></defs>`;
    for(let c=0;c<7;c++) s+=`<text x="${cx(c)}" y="17" text-anchor="middle" font-size="12" fill="#64748b">${c}</text>`;
    for(let r=0;r<5;r++) {
      s+=`<text x="24" y="${cy(r)+4}" text-anchor="middle" font-size="12" fill="#64748b">${r}</text>`;
      for(let c=0;c<7;c++) {
        const key=data.map[r][c];
        let fill=({'#':'#34445a','X':'#f6c6bf','S':'#dbeafe','P':'#fce7a1','D':'#b5ecd0'})[key]||'#f8fafc';
        const value=list[r*7+c];
        if(key!=='#' && key!=='X' && (values || world)) {
          const intensity=Math.max(0,Math.min(1,Math.abs(value||0)/max));
          fill=(value||0)<0 ? `hsl(12 64% ${98-intensity*35}%)` : `hsl(214 76% ${98-intensity*35}%)`;
        }
        s+=`<rect x="${ox+c*cell}" y="${oy+r*cell}" width="${cell}" height="${cell}" rx="3" fill="${fill}" stroke="#d5dfe9"><title>行 ${r} 列 ${c}${Number.isFinite(value)?`, ${world?'概率':'价值'} ${fmt(value)}`:''}</title></rect>`;
        if('SPDX'.includes(key)) s+=`<text x="${cx(c)}" y="${cy(r)+6}" text-anchor="middle" font-size="21" font-weight="600" fill="#24384c">${key}</text>`;
        if((values||world) && key!=='#' && key!=='X') s+=`<text x="${cx(c)}" y="${cy(r)+24}" text-anchor="middle" font-size="11" fill="#334155">${world?(value>.003?percent(value):''):fmt(value,2)}</text>`;
      }
    }
    for(const f of route) {
      if(f.position[2]!==phase) continue;
      const [r,c]=f.position,[nr,nc]=f.next_position;
      if(r===nr&&c===nc) continue;
      s+=`<line x1="${cx(c)}" y1="${cy(r)}" x2="${cx(nc)}" y2="${cy(nr)}" stroke="${phase?'#7c3aed':'#2563eb'}" stroke-width="3" opacity=".65" marker-end="url(#arrow-${id})"/>`;
    }
    if(previous) s+=`<circle cx="${cx(previous[1])}" cy="${cy(previous[0])}" r="20" fill="none" stroke="#516b8b" stroke-width="2" stroke-dasharray="4 3"/>`;
    if(predicted) s+=`<rect x="${ox+predicted[1]*cell+5}" y="${oy+predicted[0]*cell+5}" width="${cell-10}" height="${cell-10}" rx="7" fill="none" stroke="#c46c17" stroke-width="3" stroke-dasharray="6 3"/>`;
    if(current) {
      if(world) s+=`<circle cx="${cx(current[1])}" cy="${cy(current[0])-6}" r="17" fill="none" stroke="#087f5b" stroke-width="4"/>`;
      else s+=`<circle cx="${cx(current[1])}" cy="${cy(current[0])}" r="16" fill="${current[2]?'#7c3aed':'#2563eb'}" stroke="#fff" stroke-width="3"/><text x="${cx(current[1])}" y="${cy(current[0])+4}" fill="#fff" text-anchor="middle" font-size="12">${current[2]?'货':'R'}</text>`;
    }
    s+='</svg>';
    $(id).innerHTML=s;
  }

  function chart(id, series, {xLabel='时间步', yLabel='', highlight=null, minY=null,maxY=null, legend=true}={}) {
    const W=760,H=280,L=64,R=20,T=20,B=48;
    const finite=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
    if(!finite.length) {$(id).innerHTML='<p class="empty">暂无对应训练记录。</p>';return;}
    const xmin=Math.min(...finite.map(p=>p[0])), xmax=Math.max(...finite.map(p=>p[0]));
    let ymin=minY===null?Math.min(0,...finite.map(p=>p[1])):minY;
    let ymax=maxY===null?Math.max(0,...finite.map(p=>p[1])):maxY;
    if(ymax===ymin) ymax=ymin+1;
    const pad=(ymax-ymin)*.06; if(minY===null)ymin-=pad;if(maxY===null)ymax+=pad;
    const x=v=>L+(v-xmin)/(xmax-xmin||1)*(W-L-R);
    const y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-B-T);
    let svg=`<svg viewBox="0 0 ${W} ${H}" aria-label="${escape(yLabel)}随${escape(xLabel)}变化"><text x="${L}" y="12" fill="#64748b" font-size="11">${escape(yLabel)}</text>`;
    for(let i=0;i<=4;i++) {
      const v=ymin+(ymax-ymin)*i/4, yy=y(v);
      svg+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#e4eaf2"/><text x="${L-9}" y="${yy+4}" text-anchor="end" fill="#64748b" font-size="11">${Math.abs(v)>=100?v.toFixed(0):v.toFixed(2)}</text>`;
      const vx=xmin+(xmax-xmin)*i/4;
      svg+=`<text x="${x(vx)}" y="${H-B+20}" text-anchor="middle" fill="#64748b" font-size="11">${Math.abs(vx)>=1000?(vx/1000).toFixed(1)+'k':vx.toFixed(xmax<=2?1:0)}</text>`;
    }
    svg+=`<text x="${(L+W-R)/2}" y="${H-5}" text-anchor="middle" fill="#64748b" font-size="11">${escape(xLabel)}</text>`;
    if(highlight!==null) svg+=`<line x1="${x(highlight)}" x2="${x(highlight)}" y1="${T}" y2="${H-B}" stroke="#64748b" stroke-dasharray="4 3"/>`;
    series.forEach((s,j)=> {
      const points=s.points.filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
      if(!points.length)return;
      const color=s.color||colors[j%colors.length];
      svg+=`<polyline points="${points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.4" ${s.dashed?'stroke-dasharray="6 4"':''}/>`;
      points.forEach(p=>{if(points.length<100)svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="2.7" fill="${color}"><title>${escape(s.name)} · ${fmt(p[0],2)}：${fmt(p[1])}</title></circle>`;});
    });
    svg+='</svg>';
    if(legend)svg+=`<div class="chart-legend">${series.map((s,j)=>`<span><i style="background:${s.color||colors[j%colors.length]}"></i>${escape(s.name)}</span>`).join('')}</div>`;
    $(id).innerHTML=svg;
  }

  function stop() {if(state.timer)clearInterval(state.timer);state.timer=null;$('play').textContent='播放';}
  function setStep(value) {state.step=Math.max(0,Math.min(selectedEpisode().frames.length-1,Number(value)));renderReplay();renderWorld();}
  function result(label,value) {return `<div class="result"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`;}

  function renderReplay() {
    const run=selectedRun(),ep=selectedEpisode(),frame=selectedFrame();
    $('timeline').max=ep.frames.length-1;$('timeline').value=state.step;
    $('step-label').textContent=`第 ${state.step+1} / ${ep.steps} 步`;
    $('previous').disabled=state.step===0;$('next').disabled=state.step===ep.steps-1;
    $('stage').textContent=frame.next_position[2]?'已取货，前往 D':'未取货，前往 P';
    drawMap('map',{current:frame.next_position,previous:frame.position,route:ep.frames.slice(0,state.step+1),phase:frame.next_position[2]});
    const moved=frame.position[0]!==frame.next_position[0]||frame.position[1]!==frame.next_position[1];
    $('transition-label').textContent=`${position(frame.position)} → 指令 ${actions[frame.action]} → ${position(frame.next_position)}${!moved?'；这一步没有移动（可能撞墙或打滑）':''}。地图显示本步执行后的状态。`;
    $('episode-summary').innerHTML=`<span>整条回放：<strong class="${ep.success?'good':'bad'}">${escape(ep.status)}</strong> · ${ep.steps} 步</span><span>当前累计原始折扣回报：${fmt(frame.base_return_so_far)}</span>`;
    const values=frame.probabilities||frame.q_values;
    $('decision-title').textContent=frame.probabilities?'Actor 怎样选择动作？':'各动作的 Q 值是多少？';
    $('decision-note').textContent=frame.probabilities?'柱长是当前模型的动作概率，蓝色是本步选中的指令。': '柱长以本状态的最小 Q 平移后比较；右侧数字是实际 Q，不能把柱长当成概率。';
    const lower=frame.probabilities?0:Math.min(...values), upper=frame.probabilities?1:Math.max(...values);
    $('action-bars').innerHTML=values.map((v,a)=>`<div class="action-row ${a===frame.action?'chosen':''}"><span>${actions[a]}${a===frame.action?' ✓':''}</span><div class="bar-track"><div class="bar-fill" style="width:${100*(v-lower)/(upper-lower||1)}%"></div></div><span class="number">${frame.probabilities?percent(v):fmt(v)}</span></div>`).join('');
    $('base-reward').textContent=signed(frame.base_reward);
    $('shaping-reward').textContent=signed(frame.reward-frame.base_reward);
    $('training-reward').textContent=signed(frame.reward);
    let detail='';
    if(Number.isFinite(frame.target)) {
      const label=frame.target_kind==='V-TD'?'动作前预期 V(s)':frame.target_kind==='SAC-soft'?'当前选中动作 min(Q1,Q2)':'当前选中动作 Q(s,a)';
      detail=result(label,fmt(frame.current_estimate));
      detail+=`<div class="formula">target = r + γ × ${frame.terminated?'0（真正终止）':'下一状态价值'}<br>= ${fmt(frame.reward)} + ${run.gamma} × ${fmt(frame.next_value)} = ${fmt(frame.target)}<br>δ = target − 当前估计 = ${signed(frame.delta)}</div>`;
      detail+=`<p class="${frame.delta>=0?'good':'bad'}">${frame.target_kind==='V-TD'?(frame.delta>=0?'执行动作后，前景相对预期提升。':'执行动作后，前景相对预期降低。'):'该转移的 Bellman 目标与当前动作价值的差。'}</p>`;
      if(frame.target_kind==='SARSA-greedy') detail+='<p class="description">此处按回放的贪心策略选择下一动作；不代表训练时带 ε 探索的 SARSA 目标。</p>';
      if(frame.target_kind==='SAC-soft') detail+=`<p class="description">下一价值对四个动作按 π 求和，并包含 −α log π。Q1=${fmt(frame.q1)}，Q2=${fmt(frame.q2)}，α=${fmt(frame.temperature)}。</p>`;
      if(Number.isFinite(frame.advantage)) detail+=result(`供 Actor 参考的${frame.lambda?' GAE':'一步 TD'} 优势`,signed(frame.advantage));
    } else detail='<p>REINFORCE 没有神经 Critic，使用完整回报及历史常数基线更新策略；此处不伪造 V 或 TD 误差。</p>';
    detail+=result(frame.return_is_complete?'本步以后的真实折扣回报':'本步以后窗口内回报（尚未真正终止）',fmt(frame.observed_return_to_go));
    $('value-calculation').innerHTML=detail;
    const series=[{name:'学习奖励 r',color:colors[0],points:ep.frames.map(f=>[f.index,f.reward])}];
    if(Number.isFinite(frame.delta))series.push({name:'TD 误差 δ',color:colors[1],points:ep.frames.map(f=>[f.index,f.delta])});
    if(Number.isFinite(frame.advantage))series.push({name:frame.lambda?'GAE 优势':'一步优势',color:colors[2],points:ep.frames.map(f=>[f.index,f.advantage])});
    chart('trajectory-chart',series,{xLabel:'时间步 t（从 0 开始）',yLabel:'奖励 / 评价信号',highlight:state.step});
    $('advantage-note').textContent=Number.isFinite(frame.advantage)?`GAE 使用本条回放和冻结 Critic 重算；γ=${run.gamma}，λ=${frame.lambda}。它不证明每个早期动作独立造成成功。`:'完整回报汇集后续奖励；没有 V-Critic 的模型不显示 GAE。';
    $('trace-table').innerHTML=ep.frames.map(f=>`<tr data-step="${f.index}" class="${f.index===state.step?'active':''}"><td>${f.index+1}</td><td>${escape(position(f.position))} → ${escape(position(f.next_position))}</td><td>${actions[f.action]}</td><td>${signed(f.base_reward)}</td><td>${signed(f.reward)}</td><td>${signed(f.delta)}</td><td>${signed(f.advantage)}</td></tr>`).join('');
    if(!state.ppoManual)$('ppo-advantage').value=Number.isFinite(frame.advantage)?frame.advantage.toFixed(2):2;
    renderPPO();
  }

  function renderPPO() {
    const a=Number($('ppo-advantage').value)||0,r=Number($('ppo-ratio').value);
    $('ratio-label').textContent=r.toFixed(2);
    const clipped=x=>Math.min(x*a,Math.max(.8,Math.min(1.2,x))*a);
    const grid=Array.from({length:81},(_,i)=>.2+i*.02);
    chart('ppo-chart',[{name:'未裁剪 ρA',color:colors[0],dashed:true,points:grid.map(x=>[x,x*a])},{name:'PPO 采用的目标',color:colors[2],points:grid.map(x=>[x,clipped(x)])}],{xLabel:'新旧动作概率比 ρ',yLabel:'要最大化的代理目标',highlight:r});
    const plateau=(a>0&&r>1.2)||(a<0&&r<.8);
    $('ppo-readout').innerHTML=`<div class="formula">未裁剪：${fmt(r*a)}<br>PPO 采用：${fmt(clipped(r))}<br>对应 Actor 损失贡献：${fmt(-clipped(r))}</div><p>${a===0?'优势为 0，这条样本不提供策略更新信号。':plateau?'已进入平台：这条样本不再奖励沿该方向继续扩大概率变化。':'这条样本仍保留改变动作概率的梯度信号。'} ε=0.2；裁剪不是概率变化的硬上限。</p>`;
  }

  function renderBellman() {
    const demo=data.bellman,u=demo.updates[state.bellman];
    $('bellman-note').textContent=demo.explanation;
    $('bellman-timeline').max=demo.updates.length-1;$('bellman-timeline').value=state.bellman;
    $('bellman-step').textContent=`教学更新 ${state.bellman+1} / ${demo.updates.length}`;
    $('bellman-previous').disabled=state.bellman===0;$('bellman-next').disabled=state.bellman===demo.updates.length-1;
    drawMap('bellman-map',{values:u.values_after,phase:Number($('phase-select').value)});
    $('bellman-calculation').innerHTML=`<p>${escape(position(u.position))} 执行 ${actions[u.action]}，到 ${escape(position(u.next_position))}</p>`+result('更新前 Q(s,a)',fmt(u.old_q))+`<div class="formula">目标 y = r + γ × ${u.terminated?'0':'max Q(s′,a′)'}<br>= ${fmt(u.reward)} + ${demo.gamma} × ${fmt(u.next_max)}<br>= ${fmt(u.target)}<br><br>δ = y − Q旧 = ${fmt(u.delta)}<br>Q新 = Q旧 + α × δ<br>= ${fmt(u.old_q)} + ${demo.alpha} × ${fmt(u.delta)}<br>= ${fmt(u.new_q)}</div>`+result('刚更新的一个 Q 表项',fmt(u.new_q))+'<p class="description">这一条只修改访问到的 (状态,动作) 表项，其他 Q 值保持原值。切换取货阶段可查看不同状态平面。</p>';
  }

  function renderTraining() {
    const metric=$('metric-select').value;
    const labels={success_rate:'送达成功率',base_discounted_return:'原始折扣回报',hazard_rate:'危险区失败率',mean_steps:'平均步数'};
    const series=data.runs.filter(r=>curveEnabled.has(r.id)).map(r=>({name:r.id,color:colors[data.runs.indexOf(r)%colors.length],points:r.learning.map(p=>[p.steps,p[metric]])}));
    chart('training-chart',series,{xLabel:'环境交互步数',yLabel:labels[metric],minY:['success_rate','hazard_rate'].includes(metric)?0:null,maxY:['success_rate','hazard_rate'].includes(metric)?1:null});
    $('results-table').innerHTML=data.runs.map(r=>{
      const report=r.report,e=report.evaluation_result||{};
      return `<tr><td>${escape(r.id)}</td><td>${escape(report.device||'CPU（既有记录）')}</td><td>${report.environment_steps??r.learning.at(-1)?.steps??'已知模型求解'}</td><td>${percent(e.success_rate)}</td><td>${percent(e.hazard_rate)}</td><td>${fmt(e.base_discounted_return)}</td></tr>`;
    }).join('');
    chart('prediction-chart',data.predictions.map((r,j)=>({name:r.id,color:colors[j],points:r.learning.map(p=>[p.steps,Math.abs(p.start_estimate-p.start_exact)])})),{xLabel:'环境交互步数',yLabel:'起点价值绝对误差',minY:0});
  }

  function renderWorld() {
    const f=selectedFrame(),world=f.world;
    $('world-step').textContent=`当前回放第 ${state.step+1} / ${selectedEpisode().steps} 步`;
    $('world-previous').disabled=state.step===0;$('world-next').disabled=state.step===selectedEpisode().steps-1;
    if(!world) {
      $('world-map').innerHTML='<p class="empty">尚无可用的世界模型。先运行 world-model 实验，再重新生成页面。</p>';
      $('world-readout').innerHTML='';
    } else {
      drawMap('world-map',{world:world.grid_probabilities,current:f.next_position,predicted:world.top[0].position,phase:f.next_position[2]});
      let invalidMass=0;
      for(let r=0;r<5;r++)for(let c=0;c<7;c++)if(data.map[r][c]==='#')invalidMass+=world.grid_probabilities[r*7+c];
      $('world-readout').innerHTML=`<p>状态 ${escape(position(f.position))}，动作 ${actions[f.action]}</p><p class="description">模型：${escape(world.model_id)}</p>`+result('真实下一状态的预测概率',percent(world.actual_next_probability))+result('预测学习奖励 / 真实学习奖励',`${fmt(world.reward)} / ${fmt(f.reward)}`)+result('预测 continuation / 真实是否继续',`${percent(world.continuation)} / ${f.terminated?'0（终止）':'1（非真正终止）'}`)+result('模型预测落到墙内的概率',percent(invalidMass))+`<p>最可能的下一状态：</p><ol class="world-list">${world.top.map(p=>`<li>${escape(position(p.position))}：${percent(p.probability)}</li>`).join('')}</ol>`;
    }
    $('world-validation').innerHTML=data.world_models.map(m=>`<div class="description"><strong>${escape(m.id)}</strong> · held-out 下一状态 NLL ${fmt(m.report.heldout_next_state_nll)} · reward RMSE ${fmt(m.report.heldout_reward_rmse)} · continuation 准确率 ${percent(m.report.heldout_continuation_accuracy)}</div>`).join('');
  }

  function selectRun(index) {
    stop();state.run=Number(index);state.step=0;state.ppoManual=false;
    const run=selectedRun();
    if(!run.sampled)state.mode='greedy';
    $('run-select').value=state.run;$('mode-select').value=state.mode;
    $('mode-select').options[1].disabled=!run.sampled;
    const e=run.report.evaluation_result||{};
    $('eval-summary').innerHTML=`<span>独立测试送达率 <strong>${percent(e.success_rate)}</strong></span><span>危险率 ${percent(e.hazard_rate)}<br>原始回报 ${fmt(e.base_discounted_return)}</span>`;
    renderReplay();renderWorld();
  }

  function showTab(name) {
    stop();state.tab=name;
    document.querySelectorAll('[role=tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===name)));
    ['replay','bellman','training','world','textbook','embodied'].forEach(t=>$(`panel-${t}`).hidden=t!==name);
  }

  $('run-select').innerHTML=data.runs.map((r,i)=>`<option value="${i}">${escape(r.id)}</option>`).join('');
  $('seed-label').textContent=`回放环境种子 ${data.seed}`;
  $('export-meta').textContent=`${data.runs.length} 份策略记录 · 本页面离线播放 · 导出时模型推理设备 ${data.inference_device}`;
  $('curve-toggles').innerHTML=data.runs.filter(r=>r.learning.length).map(r=>`<label><input type="checkbox" checked data-curve="${escape(r.id)}"><span style="color:${colors[data.runs.indexOf(r)%colors.length]}">${escape(r.id)}</span></label>`).join('');
  $('run-select').addEventListener('change',e=>selectRun(e.target.value));
  $('mode-select').addEventListener('change',e=>{stop();state.mode=e.target.value;state.step=0;renderReplay();renderWorld();});
  $('previous').addEventListener('click',()=>{stop();setStep(state.step-1);});
  $('next').addEventListener('click',()=>{stop();setStep(state.step+1);});
  $('timeline').addEventListener('input',e=>{stop();setStep(e.target.value);});
  $('play').addEventListener('click',()=>{
    if(state.timer){stop();return;}
    if(state.step===selectedEpisode().steps-1)setStep(0);
    $('play').textContent='暂停';
    state.timer=setInterval(()=>{if(state.step>=selectedEpisode().steps-1){stop();return;}setStep(state.step+1);},Number($('speed').value));
  });
  $('speed').addEventListener('change',()=>{if(state.timer){stop();$('play').click();}});
  $('ppo-advantage').addEventListener('input',()=>{state.ppoManual=true;renderPPO();});
  $('ppo-ratio').addEventListener('input',renderPPO);
  $('trace-table').addEventListener('click',e=>{const row=e.target.closest('[data-step]');if(row){stop();setStep(row.dataset.step);}});
  function setBellman(n){state.bellman=Math.max(0,Math.min(data.bellman.updates.length-1,Number(n)));renderBellman();}
  $('bellman-previous').addEventListener('click',()=>setBellman(state.bellman-1));
  $('bellman-next').addEventListener('click',()=>setBellman(state.bellman+1));
  $('bellman-timeline').addEventListener('input',e=>setBellman(e.target.value));
  $('phase-select').addEventListener('change',renderBellman);
  $('metric-select').addEventListener('change',renderTraining);
  $('curve-toggles').addEventListener('change',e=>{const id=e.target.dataset.curve;if(id){e.target.checked?curveEnabled.add(id):curveEnabled.delete(id);renderTraining();}});
  $('world-previous').addEventListener('click',()=>setStep(state.step-1));
  $('world-next').addEventListener('click',()=>setStep(state.step+1));
  document.querySelectorAll('[role=tab]').forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.tab)));
  window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||!['replay','world'].includes(state.tab))return;if(e.key==='ArrowRight'){stop();setStep(state.step+1);}if(e.key==='ArrowLeft'){stop();setStep(state.step-1);}});
  // Small read-only testing API; no network or runtime backend is required.
  window.RL_APP={getState:()=>({run:selectedRun().id,step:state.step,mode:state.mode,tab:state.tab,playing:Boolean(state.timer)}),selectRun,setStep,showTab,chart};
  selectRun(state.run);renderBellman();renderTraining();renderWorld();
})();
