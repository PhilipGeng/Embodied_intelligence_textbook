(() => {
  'use strict';
  const data=window.RL_DATA, $=id=>document.getElementById(id), api=window.RL_APP;
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=(x,n=3)=>Number.isFinite(x)?(Math.abs(x)>0&&Math.abs(x)<.001?x.toExponential(2):x.toFixed(n)):'—';
  const pct=x=>Number.isFinite(x)?(100*x).toFixed(1)+'%':'—';
  const embodied=data.embodied||{runs:[],capabilities:[],frameworks:{}};
  const lessons=data.lessons||[];
  let lessonIndex=0, nativeIndex=0, nativeStep=0, nativeTimer=null;
  const nativeRun=()=>embodied.runs[nativeIndex];
  const route=()=>nativeRun()?.trajectories[$('domain-select').value]||[];
  const lesson=()=>lessons[lessonIndex];
  const cell=(label,value)=>`<div class="result"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
  const formula=text=>`<div class="formula">${esc(text).replace(/\n/g,'<br>')}</div>`;
  const scalar=(name,fallback=0)=>{const input=$('lab-'+name);return input?Number(input.value):fallback;};
  const slider=(name,label,min,max,step,value)=>`<label>${esc(label)}：<output id="lab-${name}-value">${num(value)}</output><input type="range" id="lab-${name}" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
  const vector=a=>'['+a.map(x=>num(x,2)).join(', ')+']';
  const ppoRun=()=>data.runs.find(r=>r.id===api.getState().run&&r.greedy.frames[0].value!==undefined)||data.runs.find(r=>r.algorithm==='ppo');

  function showSource(variable=null) {
    const source=lesson().code;
    let hits=0;
    $('lesson-source').innerHTML=source.lines.map(line=>{
      let text=esc(line.text),active=variable&&line.text.includes(variable);
      if(active){hits++;text=text.split(esc(variable)).join('<mark>'+esc(variable)+'</mark>');}
      return `<div class="code-line ${active?'is-highlighted':''}"><span class="code-number">${line.number}</span><span>${text||' '}</span></div>`;
    }).join('');
    $('lesson-source-link').textContent=source.path+':'+source.line;
    $('lesson-source-link').href='../'+source.path;
    $('lesson-code-note').textContent=variable?(hits?`${variable} 出现在 ${hits} 行；这是导出时从项目源文件读取的内容。`:`${variable} 的完整定义请查看源文件；当前节选中没有这个变量。`):'真实源代码节选，行号来自当前项目；点击左侧变量可高亮对应位置。';
  }

  function showEvidence() {
    const l=lesson(), mapping={bellman:'bellman',sarsa:'sarsa',q:'q',dqn:'dqn',reinforce:'reinforce',actor:'ac',gae:'gae',ppo:'ppo','discrete-sac':'sac'};
    let html='';
    if(l.id==='mc'||l.id==='td') {
      const record=data.predictions.find(r=>r.report.algorithm===l.id),e=record?.report.prediction_error;
      if(e)html=`固定策略起点：估计 ${num(e.start_estimate)}；精确期望 ${num(e.start_exact)}；绝对误差 ${num(Math.abs(e.start_estimate-e.start_exact))}。`;
    } else if(mapping[l.id]) {
      const record=data.runs.find(r=>r.algorithm===mapping[l.id]),e=record?.report.evaluation_result;
      if(e)html=`实际记录 ${esc(api.algorithmLabel(record))}：独立评估送达率 ${pct(e.success_rate)}，原始折扣回报 ${num(e.base_discounted_return)}。`; 
    } else {
      let records=embodied.runs.filter(r=>r.report.algorithm===(l.id==='randomization'?'sac':l.id));
      if(l.id==='sac')records=records.filter(r=>!r.report.sparse&&!r.report.randomized_training);
      if(l.id==='randomization')records=records.filter(r=>!r.report.sparse);
      if(records.length)html=records.map(r=>{
        const n=r.report.evaluation?.nominal,s=r.report.evaluation?.shifted;
        return `<p><strong>${esc(api.algorithmLabel(r))}</strong><br>名义最终成功率 ${pct(n?.success_rate)}，偏移域 ${pct(s?.success_rate)}；名义最终距离 ${num(n?.final_distance)}。${r.report.dataset_hash?'<br>固定数据 hash：'+esc(r.report.dataset_hash.slice(0,16))+'…':''}${r.report.dataset_coverage?'<br>覆盖：'+esc(r.report.dataset_coverage):''}</p>`;
      }).join('');
    }
    if(l.id==='world-model')html=data.world_models.map(r=>`<p>独立验证：下一状态 NLL ${num(r.report.heldout_next_state_nll)}，奖励 RMSE ${num(r.report.heldout_reward_rmse)}。</p>`).join('');
    if(['memory','frameworks'].includes(l.id))html=`已有 ${(embodied.frameworks.smoke_experiments||[]).length} 个第三方短跑记录。到“具身进阶”查看框架、环境与预算；短跑只验证代码流程，不证明收敛。`;
    $('lesson-evidence').innerHTML=`<div class="lesson-evidence">${html||'当前没有这一课的对应运行结果。先执行上方命令，保存报告后重新生成页面。'}<p>这里展示真实保存的实验。单种子结果不构成算法优劣结论，简化仿真也不等于真机性能。</p></div>`;
  }


  function renderTheory(l) {
    const t=l.theory;
    if(!t){$('lesson-theory').textContent='本章详细理论未载入，请重新导出教材。';return;}
    const paragraphs=items=>items.map(x=>`<p>${esc(x)}</p>`).join('');
    $('lesson-theory').innerHTML=`
      <div class="theory-intro"><h3>本章依赖什么基础？</h3><p>${esc(t.prerequisite)}</p></div>
      <h3>相比之前，为什么需要这项技术？</h3>
      <div class="theory-evolution">
        ${[['原有方法的瓶颈',t.weakness],['核心改动',t.change],['获得的新能力',t.capability]].map(([label,value])=>`<div><h4>${esc(label)}</h4><p>${esc(value)}</p></div>`).join('')}
      </div>
      <h3>设计动机与关键假设</h3>${paragraphs(t.design)}
      <details class="theory-notation"><summary>展开共同符号约定：状态、奖励、终止、期望与梯度</summary><dl>${l.notation.map(([symbol,meaning])=>`<dt>${esc(symbol)}</dt><dd>${esc(meaning)}</dd>`).join('')}</dl></details>
      <h3>逐个公式推导，逐项解释</h3>
      <div class="theory-equations">${t.equations.map((e,i)=>`<article class="theory-equation"><h4>公式 ${l.chapter}.${i+1} · ${esc(e.title)}</h4><pre class="lesson-formula">${esc(e.formula)}</pre><h5>公式里的每一项</h5><dl class="theory-symbols">${e.symbols.map(term=>`<dt>${esc(term.symbol)}</dt><dd>${esc(term.meaning)}</dd>`).join('')}</dl><h5>这一步是怎样得到的？</h5><ol>${e.steps.map(step=>`<li>${esc(step)}</li>`).join('')}</ol><p class="theory-purpose"><strong>为什么这样写：</strong>${esc(e.why)}</p></article>`).join('')}</div>
      <h3>代入数字，检查是否真的理解</h3><p class="theory-example">${esc(t.example)}</p>
      <h3>公式怎样组成实际算法？</h3><ol class="theory-workflow">${t.workflow.map(step=>`<li>${esc(step)}</li>`).join('')}</ol>
      <h3>仍未解决什么？</h3><p class="theory-limitation">${esc(t.limitation)}</p>
      <p class="description">下面将本章公式映射到项目源码；继续操作参数实验，再运行实际训练命令核对。</p><hr>`;
  }

  function renderLesson() {
    const l=lesson();if(!l)return;
    $('lesson-title').textContent=l.title;
    $('lesson-select').value=lessonIndex;
    $('lesson-previous').disabled=lessonIndex===0;
    $('lesson-next').disabled=lessonIndex===lessons.length-1;
    const before=lessons[lessonIndex-1],after=lessons[lessonIndex+1];
    $('lesson-connection').textContent=(before?'承接 '+before.title+'。':'从回报的定义开始，建立价值递推。')+' 本章问题：'+l.problem+(after?' 接下来：'+after.title+'。':'接下来可按框架接入路线扩展实验。');
    $('lesson-problem').textContent=l.problem;
    renderTheory(l);
    $('lesson-derivation').innerHTML=l.derivation.map(x=>'<li>'+esc(x)+'</li>').join('');
    $('lesson-formula').textContent=l.formula;
    $('lesson-terms').innerHTML=l.terms.map(t=>`<tr><td>${esc(t.symbol)}</td><td><button class="term-key" data-variable="${esc(t.variable)}">${esc(t.variable)}</button></td><td>${esc(t.meaning)}</td></tr>`).join('');
    $('lesson-command').textContent=l.command;
    $('lesson-observe').textContent=l.observe;
    $('lesson-question').textContent=l.quiz.question;
    $('lesson-choices').innerHTML=l.quiz.choices.map((choice,i)=>`<label class="lesson-choice"><input type="radio" name="lesson-answer" value="${i}"><span>${esc(choice)}</span></label>`).join('');
    $('lesson-feedback').textContent='';showSource();showEvidence();
    let controls='',note='';
    if(l.lab==='bellman') {
      controls=slider('gamma','折扣 γ',0,.99,.01,.97)+slider('alpha','学习率 α',.01,1,.01,.5);
      note='固定一条教学回放的奖励、旧 Q 与下一状态估值，改变折扣或步长，观察局部更新。塑形奖励也先固定，不等于重训整个环境。';
    } else if(l.lab==='gae') {
      controls=slider('gamma','折扣 γ',0,.99,.01,.97)+slider('lambda','GAE λ',0,1,.01,l.id==='actor'?0:.95);
      note='使用已训练 Critic 和真实回放奖励重算；固定预测值，查看多步误差怎样组合，不更新网络。';
    } else if(l.lab==='reinforce') {
      controls=slider('return','完整回报 G',-10,10,.1,6)+slider('baseline','历史基线 b',-5,5,.1,2);
      note='固定当前动作概率 π=0.25，查看完整回报相对历史基线如何改变 logπ 的梯度系数。这是单步演算，真实训练按完整回合求和。';
    } else if(l.lab==='ppo') {
      controls=slider('adv','优势 A',-5,5,.01,2)+slider('ratio','新旧概率比 ρ',.2,1.8,.01,1);
      note='单样本公式实验：优势与比率是你设定的数，不伪装成历史更新，也不构造完整的新动作分布。';
    } else if(l.lab==='sac') {
      controls=slider('alpha','熵温度 α',0,1,.01,l.id==='td3'?0:.2)+slider('q2','第二个 Critic 的 Q',-8,6,.1,-4);
      note='明确标注的标量教学样本：r=-.2、γ=.98、Q1=-3、logπ=-.7。下方真实训练结果与这个数字演示分开。';
    } else if(l.lab==='her') {
      const reference=embodied.runs.find(r=>r.report.algorithm==='pd')||embodied.runs[0];
      const count=reference?.trajectories.nominal?.length||60;
      controls=slider('step','真实轨迹时间步',0,count-1,1,10)+slider('future','选哪个未来达到位置作目标',0,count-1,1,30);
      note='取保存的模拟轨迹，保持当前状态、动作、下一状态不变，分别对原目标与事后目标重算同一个稀疏奖励函数。';
    } else if(l.lab==='bc') {
      controls=slider('demo','演示动作 a_demo',-1,1,.01,.8)+slider('prediction','预测动作 μ',-1,1,.01,.2)+slider('alpha','标量学习率',.01,.5,.01,.1);
      note='将 μ 暂时视为单个可训练标量，演示平方损失和一次梯度下降；实际代码还通过网络的链式法则更新参数。';
    } else if(l.lab==='iql') {
      controls=slider('tau','Expectile τ',.5,.95,.01,.7)+slider('beta','优势权重 β',.1,3,.1,.5);
      note='同一状态下两个等频数据动作 Q=2 和 8 的手算例子；显示 expectile 与模仿权重，避免把演示数字当成训练值。';
    } else if(l.lab==='mpc') {
      controls=slider('horizon','预测步数 H',1,25,1,15)+slider('step','从实际轨迹哪一步规划',0,59,1,20);
      note='使用真正拟合保存的动力学权重。页面只比较九个恒定力候选，Python 实践用 128 候选 CEM；不要把两者的优化精度混同。';
    } else if(l.lab==='dynamics') {
      controls=slider('mass','质量 m',.6,1.8,.05,1)+slider('drag','阻尼',.1,.8,.05,.35)+slider('delay','动作延迟步数',0,3,1,0);
      note='严格按环境里的半隐式积分与延迟队列模拟同一个固定 PD 控制器；改变的是物理条件，不是在浏览器里训练 SAC。';
    } else note=l.lab==='prediction'?'右侧与训练曲线中的 MC/TD 是同一固定策略。先观察估值误差，再改变回合预算重新运行。':'这一课用实际外部框架实验与可运行命令验证。查看“具身进阶”的安装状态与短跑报告。';
    $('lesson-lab-controls').innerHTML=controls;$('lesson-lab-note').textContent=note;
    updateLab();
  }

  function drawContinuous(id,frames,index,goalOverride=null) {
    if(!frames.length){$(id).innerHTML='<p class="empty-topic">暂无对应轨迹。</p>';return;}
    index=Math.max(0,Math.min(index,frames.length-1));
    const W=500,H=420,ox=50,oy=24,S=350;
    const x=v=>ox+(v+1)*S/2,y=v=>oy+(1-v)*S/2;
    const f=frames[index],goal=goalOverride||f.goal;
    let svg=`<svg viewBox="0 0 ${W} ${H}" aria-label="连续平面中的真实轨迹与目标"><rect x="${ox}" y="${oy}" width="${S}" height="${S}" rx="8" fill="#f5f8fc" stroke="#ced9e7"/>`;
    for(const v of [-1,-.5,0,.5,1])svg+=`<line x1="${x(v)}" x2="${x(v)}" y1="${oy}" y2="${oy+S}" stroke="#e2e9f2"/><line x1="${ox}" x2="${ox+S}" y1="${y(v)}" y2="${y(v)}" stroke="#e2e9f2"/><text x="${x(v)}" y="${oy+S+21}" text-anchor="middle" fill="#64748b" font-size="11">${v}</text><text x="${ox-12}" y="${y(v)+4}" text-anchor="end" fill="#64748b" font-size="11">${v}</text>`;
    const points=[frames[0].position,...frames.slice(0,index+1).map(f=>f.next_position)];
    svg+=`<polyline points="${points.map(p=>x(p[0])+','+y(p[1])).join(' ')}" fill="none" stroke="#2563eb" stroke-width="2.5"/><circle cx="${x(goal[0])}" cy="${y(goal[1])}" r="${S/2*.12}" fill="#d6f2e6" stroke="#087f5b" stroke-width="2"/><text x="${x(goal[0])+15}" y="${y(goal[1])-15}" fill="#087f5b" font-size="12">${goalOverride?'g′':'目标'}</text><circle cx="${x(f.next_position[0])}" cy="${y(f.next_position[1])}" r="8" fill="#2563eb" stroke="#fff" stroke-width="2"/><text x="${ox+S/2}" y="${H-5}" text-anchor="middle" fill="#64748b" font-size="11">位置 x（动作是力，不是位置增量）</text></svg>`;
    $(id).innerHTML=svg;
  }

  function updateLab() {
    const l=lesson();if(!l)return;
    $('lesson-lab-controls').querySelectorAll('input[type=range]').forEach(input=>{const output=$(input.id+'-value');if(output)output.textContent=num(Number(input.value));});
    const host=$('lesson-lab-result'),chart=$('lesson-lab-chart');chart.innerHTML='';
    if(l.lab==='bellman') {
      const u=data.bellman.updates[1],g=scalar('gamma'),a=scalar('alpha');
      const target=u.reward+g*u.next_max,delta=target-u.old_q,newQ=u.old_q+a*delta;
      host.innerHTML=formula(`y = ${num(u.reward)} + ${num(g)} × ${num(u.next_max)} = ${num(target)}\nδ = y − ${num(u.old_q)} = ${num(delta)}\nQ_new = ${num(u.old_q)} + ${num(a)} × ${num(delta)} = ${num(newQ)}`)+cell('本次 Q 改变量',num(newQ-u.old_q));
      api.chart('lesson-lab-chart',[{name:'同一目标下，不同步长的新 Q',points:Array.from({length:21},(_,i)=>[i/20,u.old_q+i/20*delta])}],{xLabel:'学习率 α',yLabel:'更新后的 Q',highlight:a});
    } else if(l.lab==='gae') {
      const run=ppoRun(),frames=run.greedy.frames,g=scalar('gamma'),lam=scalar('lambda');
      const deltas=frames.map(f=>f.reward+g*f.next_value-f.value),advantages=[];
      let last=0;for(let i=frames.length-1;i>=0;i--){last=deltas[i]+g*lam*(!(frames[i].terminated||frames[i].truncated))*last;advantages[i]=last;}
      host.innerHTML=cell('真实回放来源',api.algorithmLabel(run))+formula(`δ_0 = r_0 + γV_1 − V_0 = ${num(deltas[0])}\nA_0 = δ_0 + γλA_1 = ${num(advantages[0])}`);
      api.chart('lesson-lab-chart',[{name:'一步 δ',points:deltas.map((v,i)=>[i,v])},{name:'多步 A',points:advantages.map((v,i)=>[i,v])}],{xLabel:'轨迹时间步',yLabel:'评价信号'});
    } else if(l.lab==='reinforce') {
      const advantage=(scalar('return')-scalar('baseline'))/10;
      host.innerHTML=formula(`A_scaled = (G−b)/10 = ${num(advantage)}\nL_t = −log(0.25) × A_scaled = ${num(-Math.log(.25)*advantage)}\n∂L_t/∂logπ = −A_scaled = ${num(-advantage)}`)+cell('更新方向',advantage>0?'增大此动作概率':advantage<0?'减小此动作概率':'本样本不推动策略');
    } else if(l.lab==='ppo') {
      const a=scalar('adv'),rho=scalar('ratio'),clip=Math.max(.8,Math.min(1.2,rho)),objective=Math.min(rho*a,clip*a);
      host.innerHTML=formula(`ρA = ${num(rho*a)}\nclip(ρ)A = ${num(clip*a)}\nPPO 目标 = min(两者) = ${num(objective)}\n对应最小化的损失 = ${num(-objective)}`);
      const xs=Array.from({length:81},(_,i)=>.2+i*.02);
      api.chart('lesson-lab-chart',[{name:'ρA',points:xs.map(x=>[x,x*a]),dashed:true},{name:'PPO 目标',points:xs.map(x=>[x,Math.min(x*a,Math.max(.8,Math.min(1.2,x))*a)])}],{xLabel:'概率比 ρ',yLabel:'代理目标',highlight:rho});
    } else if(l.lab==='sac') {
      const alpha=l.id==='td3'?0:scalar('alpha'),q2=scalar('q2'),q=Math.min(-3,q2),future=q-alpha*(-.7),target=-.2+.98*future;
      host.innerHTML=formula(`min(Q1,Q2) = min(-3, ${num(q2)}) = ${num(q)}\n未来值 = ${num(q)} − ${num(alpha)} × (-0.7) = ${num(future)}\ny = -0.2 + .98 × ${num(future)} = ${num(target)}`)+cell(l.id==='td3'?'TD3 不加入熵项':'α 对未来价值的增加量',num(alpha*.7));
    } else if(l.lab==='her') {
      const reference=embodied.runs.find(r=>r.report.algorithm==='pd')||embodied.runs[0],frames=reference?.trajectories.nominal||[];
      if(!frames.length){host.innerHTML='先运行连续 PD 或 HER 示例以获得真实轨迹。';return;}
      const t=Math.min(frames.length-1,scalar('step')),input=$('lab-future');input.min=t;
      if(Number(input.value)<t)input.value=t;
      const future=Math.min(frames.length-1,Number(input.value));$('lab-future-value').textContent=String(future);
      const f=frames[t],newGoal=frames[future].next_position;
      const distance=Math.hypot(f.next_position[0]-f.goal[0],f.next_position[1]-f.goal[1]);
      const newDistance=Math.hypot(f.next_position[0]-newGoal[0],f.next_position[1]-newGoal[1]);
      host.innerHTML=formula(`真实转移：${vector(f.position)} → ${vector(f.next_position)}\n动作保持：${vector(f.action)}\n原目标：${vector(f.goal)}；稀疏奖励 ${distance<.12?0:-1}\n事后目标：第 ${future} 步到达的 ${vector(newGoal)}\n到新目标距离 ${num(newDistance)}；重算奖励 ${newDistance<.12?0:-1}`);
      drawContinuous('lesson-lab-chart',frames,t,newGoal);
    } else if(l.lab==='bc') {
      const a=scalar('demo'),mu=scalar('prediction'),lr=scalar('alpha'),loss=(mu-a)**2,gradient=2*(mu-a);
      host.innerHTML=formula(`L = (μ − a_demo)² = ${num(loss)}\n∂L/∂μ = 2(μ−a_demo) = ${num(gradient)}\nμ_new = μ − α∂L/∂μ = ${num(mu-lr*gradient)}`)+cell('朝演示动作移动了多少',num(-lr*gradient));
    } else if(l.lab==='iql') {
      const tau=scalar('tau'),beta=scalar('beta'),v=tau*8+(1-tau)*2;
      const low=Math.min(100,Math.exp(beta*(2-v))),high=Math.min(100,Math.exp(beta*(8-v)));
      host.innerHTML=formula(`等频 Q 为 2 与 8：\n(1−τ)(V−2) = τ(8−V)\nV = ${num(v)}\n较差动作 A = ${num(2-v)}，权重 = ${num(low)}\n较好动作 A = ${num(8-v)}，权重 = ${num(high)}\nBC 则两个样本等权（均为 1）`)+cell('IQL 两个模仿权重之比',num(high/low));
    } else if(l.lab==='mpc') {
      const record=embodied.runs.find(r=>r.dynamics_weights),frames=record?.trajectories.nominal||[];
      if(!record){host.innerHTML='先运行 MPC 训练并保存动力学模型。';return;}
      const index=Math.min(frames.length-1,scalar('step')),f=frames[index],h=scalar('horizon'),w=record.dynamics_weights;
      const candidates=[];
      for(const ux of [-1,0,1])for(const uy of [-1,0,1]){
        let s=[...f.position,...f.velocity],cost=0;const trajectory=[];
        for(let t=0;t<h;t++){
          const features=[...s,ux,uy,1],next=s.map((x,j)=>x+features.reduce((sum,a,i)=>sum+a*w[i][j],0));
          for(let j=0;j<4;j++)next[j]=Math.max(j<2?-1:-1.5,Math.min(j<2?1:1.5,next[j]));
          cost+=.98**t*((next[0]-f.goal[0])**2+(next[1]-f.goal[1])**2+.05*(next[2]**2+next[3]**2)+.01*(ux*ux+uy*uy));
          trajectory.push({position:s.slice(0,2),next_position:next.slice(0,2),goal:f.goal});s=next;
        }
        cost+=3*((s[0]-f.goal[0])**2+(s[1]-f.goal[1])**2);candidates.push({action:[ux,uy],cost,trajectory});
      }
      candidates.sort((a,b)=>a.cost-b.cost);
      host.innerHTML=cell('简化规划选择的恒定力',vector(candidates[0].action))+cell('预测总成本',num(candidates[0].cost))+`<p class="description">模型来自 ${esc(api.algorithmLabel(record))}，留出状态 RMSE ${num(record.report.model_validation.heldout_state_rmse)}。这是模型预测图，尚未执行到环境。</p>`;
      drawContinuous('lesson-lab-chart',candidates[0].trajectory,h-1);
    } else if(l.lab==='dynamics') {
      const mass=scalar('mass'),drag=scalar('drag'),delay=scalar('delay'),goal=[.7,.7],queue=Array.from({length:delay},()=>[0,0]);
      let p=[-.7,-.7],v=[0,0];const frames=[];
      for(let t=0;t<60;t++){
        const action=p.map((x,j)=>Math.max(-1,Math.min(1,3*(goal[j]-x)-2*v[j])));queue.push(action);const applied=queue.shift();
        const nv=v.map((x,j)=>Math.max(-1.5,Math.min(1.5,x+.12*(applied[j]/mass-drag*x))));
        const proposed=p.map((x,j)=>x+.12*nv[j]),next=proposed.map(x=>Math.max(-1,Math.min(1,x)));
        for(let j=0;j<2;j++)if(Math.abs(proposed[j])>1)nv[j]=0;
        frames.push({position:p,velocity:v,action,next_position:next,goal});p=next;v=nv;
      }
      host.innerHTML=formula(`v′ = clip[v + .12(u/${num(mass)} − ${num(drag)}v)]\np′ = clip[p + .12v′]\n延迟队列：${delay} 步`)+cell('固定 PD 控制器的最终距离',num(Math.hypot(p[0]-.7,p[1]-.7)));
      drawContinuous('lesson-lab-chart',frames,59);
    } else if(l.lab==='prediction') {
      const records=data.predictions.map(r=>({name:api.algorithmLabel(r),points:r.learning.map(x=>[x.steps,Math.abs(x.start_estimate-x.start_exact)])}));
      host.innerHTML='<p>这些点来自已经执行并保存的 MC/TD 实验，不是根据示意公式编造的曲线。</p>';
      api.chart('lesson-lab-chart',records,{xLabel:'环境交互步数',yLabel:'起点估值绝对误差',minY:0});
    } else host.innerHTML='<p>请使用实际训练命令，或到“具身进阶”检查第三方依赖和短跑结果。未安装的框架会明确显示未安装。</p>';
  }

  function stopNative(){if(nativeTimer)clearInterval(nativeTimer);nativeTimer=null;$('embodied-play').textContent='播放';}
  function renderNative() {
    const run=nativeRun(),frames=route();
    if(!run||!frames.length){$('continuous-map').innerHTML='<p class="empty-topic">先运行 run_embodied.py 生成连续控制实验。</p>';return;}
    nativeStep=Math.max(0,Math.min(nativeStep,frames.length-1));const f=frames[nativeStep],domain=$('domain-select').value,e=run.report.evaluation[domain];
    $('embodied-timeline').max=frames.length-1;$('embodied-timeline').value=nativeStep;
    $('embodied-step').textContent=`第 ${nativeStep+1} / ${frames.length} 步`;
    $('embodied-previous').disabled=nativeStep===0;$('embodied-next').disabled=nativeStep===frames.length-1;
    drawContinuous('continuous-map',frames,nativeStep);
    const bars=f.action.map((a,j)=>`<div class="force-row"><span>力 ${j?'y':'x'}</span><div class="force-track"><div class="force-value" style="left:${a<0?(50+a*50):50}%;width:${Math.abs(a)*50}%"></div></div><span class="number">${num(a)}</span></div>`).join('');
    $('continuous-readout').innerHTML=cell('完整评估最终成功率',pct(e.success_rate))+cell('曾到达率 / 最终平均距离',pct(e.ever_reached_rate)+' / '+num(e.final_distance))+formula(`位置 ${vector(f.position)}\n速度 ${vector(f.velocity)}\n目标 ${vector(f.goal)}`)+bars+cell('本步奖励 / 执行后距离',num(f.reward)+' / '+num(f.distance))+`<p class="description">${esc(run.report.scope)}。${run.report.dataset_coverage?'数据覆盖：'+esc(run.report.dataset_coverage)+'。':''}原始数据集大小 ${run.report.dataset_transitions??'不适用'}；在线训练交互 ${run.report.training_environment_steps??0} 步。</p>`;
    if(run.learning.length)api.chart('continuous-learning-chart',[{name:'名义最终成功率',points:run.learning.map(x=>[x.steps??x.updates,x.success_rate])}],{xLabel:run.learning[0].steps?'真实环境交互步数':'固定数据梯度更新次数',yLabel:'最终成功率',minY:0,maxY:1});
    else $('continuous-learning-chart').innerHTML='<p class="description">这个条目没有策略梯度学习曲线：PD 为手工参考，MPC 先拟合模型再在线规划。</p>';
  }

  function renderCatalog() {
    $('capability-table').innerHTML=embodied.capabilities.map(c=>`<tr><td>${esc(c.title)}</td><td><span class="capability-status">${esc(c.status)}</span></td><td class="wrap">${esc(c.detail)}</td><td>${esc(c.command)}</td></tr>`).join('');
    const frameworks=embodied.frameworks.frameworks||[];
    $('framework-table').innerHTML=frameworks.length?`<div class="table-wrap"><table><thead><tr><th>框架</th><th>本机检测</th><th>版本</th><th>职责</th></tr></thead><tbody>${frameworks.map(f=>`<tr><td><a href="${esc(f.official)}" target="_blank" rel="noopener">${esc(f.name)}</a></td><td>${esc(f.status)}</td><td>${esc(f.version||'—')}</td><td>${esc(f.purpose)}</td></tr>`).join('')}</tbody></table></div>`:'<p>运行 framework_doctor.py 后重新导出，可读取本机真实依赖状态。</p>';
    $('framework-experiments').innerHTML=(embodied.frameworks.smoke_experiments||[]).map(r=>`<div class="framework-card"><strong>${esc(api.algorithmLabel(r))}</strong> · ${esc(r.environment)} · ${r.actual_steps} 步 · ${esc(r.device)} · 平均回报 ${num(r.mean_return)}。这条记录仅验证真实训练流程，预算不足以证明收敛。</div>`).join('');
  }

  $('lesson-select').innerHTML=lessons.map((l,i)=>`<option value="${i}">${esc(l.title)}</option>`).join('');
  function selectLesson(id,sync=true) {
    const i=lessons.findIndex(l=>l.id===id);if(i<0)return;
    lessonIndex=i;renderLesson();
    if(sync){
      const runIndex=data.runs.findIndex(r=>api.lessonKey(r)===id);
      if(runIndex>=0&&api.getState().run!==data.runs[runIndex].id)api.selectRun(runIndex);
      const native=embodied.runs.findIndex(r=>api.lessonKey(r)===id);
      if(native>=0){stopNative();nativeIndex=native;nativeStep=0;$('embodied-select').value=native;renderNative();}
    }
  }
  $('lesson-select').addEventListener('change',e=>selectLesson(lessons[Number(e.target.value)].id));
  $('lesson-previous').addEventListener('click',()=>{if(lessonIndex>0)selectLesson(lessons[lessonIndex-1].id);});
  $('lesson-next').addEventListener('click',()=>{if(lessonIndex<lessons.length-1)selectLesson(lessons[lessonIndex+1].id);});
  window.addEventListener('rl-run-changed',e=>selectLesson(api.lessonKey(e.detail.run),false));
  $('lesson-terms').addEventListener('click',e=>{const b=e.target.closest('[data-variable]');if(b)showSource(b.dataset.variable);});
  $('lesson-lab-controls').addEventListener('input',updateLab);
  $('lesson-check').addEventListener('click',()=>{const selected=document.querySelector('input[name=lesson-answer]:checked');if(!selected){$('lesson-feedback').textContent='先选择一个答案。';return;}const correct=Number(selected.value)===lesson().quiz.answer;$('lesson-feedback').className=correct?'good':'bad';$('lesson-feedback').textContent=(correct?'理解正确。':'再想一下。')+lesson().quiz.explanation;});
  $('run-select').addEventListener('change',()=>{if(lesson())updateLab();});
  $('embodied-select').innerHTML=embodied.runs.map((r,i)=>`<option value="${i}">${esc(api.chapterLabel(r))}</option>`).join('');
  $('embodied-select').addEventListener('change',e=>{stopNative();nativeIndex=Number(e.target.value);nativeStep=0;renderNative();selectLesson(api.lessonKey(nativeRun()),false);});
  $('domain-select').addEventListener('change',()=>{stopNative();nativeStep=0;renderNative();});
  $('embodied-timeline').addEventListener('input',e=>{stopNative();nativeStep=Number(e.target.value);renderNative();});
  $('embodied-previous').addEventListener('click',()=>{stopNative();nativeStep--;renderNative();});
  $('embodied-next').addEventListener('click',()=>{stopNative();nativeStep++;renderNative();});
  $('embodied-play').addEventListener('click',()=>{if(nativeTimer){stopNative();return;}if(nativeStep>=route().length-1)nativeStep=0;$('embodied-play').textContent='暂停';nativeTimer=setInterval(()=>{if(nativeStep>=route().length-1){stopNative();return;}nativeStep++;renderNative();},350);});
  document.querySelectorAll('[role=tab]').forEach(b=>b.addEventListener('click',stopNative));
  window.TEXTBOOK_APP={getLesson:()=>lesson()?.id,selectLesson,getNative:()=>({id:nativeRun()?.id,step:nativeStep,domain:$('domain-select').value})};
  selectLesson(api.lessonKey(data.runs.find(r=>r.id===api.getState().run)),false);renderNative();renderCatalog();
})();
