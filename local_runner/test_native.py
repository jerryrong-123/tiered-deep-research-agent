"""Offline checks of the actual provider/stream boundary and native workflow."""
import asyncio, json, os, re, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import httpx
import run_native as r

class Checks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.old_here=r.HERE;self.old_helper_here=r.helpers.HERE
        r.HERE=Path(self.tmp.name);r.helpers.HERE=r.HERE
        self.old_send=httpx.AsyncClient.send
    def tearDown(self):
        httpx.AsyncClient.send=self.old_send;r.HERE=self.old_here;r.helpers.HERE=self.old_helper_here;self.tmp.cleanup()
    def test_limits_and_reasoning_roundtrip(self):
        p={'model':'deepseek-flash','max_completion_tokens':32768,'messages':[{'role':'assistant','reasoning_content':'private-memory-only','tool_calls':[]}],
            'thinking':{'type':'enabled'},'reasoning_effort':'high'}
        v=r.provider_payload(p)
        self.assertEqual(v['max_tokens'],32768);self.assertNotIn('max_completion_tokens',v)
        self.assertEqual(v['messages'],p['messages']);self.assertEqual(v['reasoning_effort'],'high')
        self.assertNotIn('private-memory-only',json.dumps(r.public(v)))
        with self.assertRaises(RuntimeError):r.provider_payload({**p,'max_tokens':10})
    async def test_stream_chunk_boundaries_usage_and_public_text(self):
        meter=r.Meter();idx=await meter.reserve({'max_tokens':100},'offline')
        frames=[{'choices':[{'index':0,'delta':{'reasoning_content':'private-memory-only'}}]},
            {'model':'deepseek-flash','choices':[{'index':0,'delta':{'content':'public answer'},'finish_reason':'stop'}]},
            {'choices':[],'usage':{'prompt_tokens':30,'completion_tokens':8,'prompt_cache_hit_tokens':10}}]
        raw=b''.join(b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames)+b'data: [DONE]\n\n'
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for i in range(0,len(raw),7):yield raw[i:i+7]
        stream=r.RecordedStream(Stream(),idx,meter,True,200)
        returned=b''.join([c async for c in stream]);await stream.aclose()
        self.assertEqual(returned,raw);self.assertEqual(meter.held,0);self.assertEqual(meter.reserved,0)
        saved=(r.HERE/'responses/00000.json').read_text()
        self.assertNotIn('private-memory-only',saved);self.assertIn('public answer',saved)
    async def test_incomplete_response_retains_unknown(self):
        meter=r.Meter();idx=await meter.reserve({'max_tokens':100},'offline')
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[]}\n\n'
                raise httpx.RemoteProtocolError('peer closed connection without sending complete message body')
        stream=r.RecordedStream(Stream(),idx,meter,True,200)
        with self.assertRaises(httpx.RemoteProtocolError):
            async for _ in stream:pass
        self.assertGreater(meter.held,0);self.assertEqual(meter.reserved,0)
    async def test_native_pipeline_and_provider_without_network(self):
        r.credentials();os.chdir(r.UP)
        meter=r.Meter();r.install_transport(meter,offline=True)
        requests=[]
        async def responder(request):
            payload=json.loads(request.content);requests.append(payload)
            self.assertEqual(payload['model'],'deepseek-flash');self.assertIn('max_tokens',payload)
            self.assertNotIn('max_completion_tokens',payload)
            if not payload.get('stream'):
                return httpx.Response(200,json={'id':'offline','object':'chat.completion','created':0,'model':'deepseek-flash',
                    'choices':[{'index':0,'message':{'role':'assistant','content':'Offline interface smoke test completed.'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'prompt_cache_hit_tokens':0}})
            frames=[{'id':'offline','object':'chat.completion.chunk','created':0,'model':'deepseek-flash',
                'choices':[{'index':0,'delta':{'role':'assistant','content':'Offline interface smoke test completed.'},'finish_reason':None}]},
                {'id':'offline','object':'chat.completion.chunk','created':0,'model':'deepseek-flash',
                'choices':[{'index':0,'delta':{},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'prompt_cache_hit_tokens':0}}]
            data=b''.join(b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames)+b'data: [DONE]\n\n'
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content=data)
        original_init=httpx.AsyncClient.__init__
        def init(client,*a,**kw):
            kw['transport']=httpx.MockTransport(responder);kw['trust_env']=False;original_init(client,*a,**kw)
        with patch.object(httpx.AsyncClient,'__init__',init):
            from benchmarks.public.core.kernel_adapter import BenchmarkSession
            from workflows.agent_team.nodes.main_agent import _llm_cache
            _llm_cache.clear()
            async with BenchmarkSession() as session:
                result=await session.run('Return a short offline interface smoke-test response.',pipeline_id='agent_team',
                    meta={'profile':'benchmark','profile_inline':r.selected_profile(),'profile_overrides':r.overrides(),'_trial_dir':str(r.HERE/'native')})
        self.assertTrue(requests)
        # This mock answers with no research. A light-stage run must refuse that
        # and stop with its marker; every other profile delivers the mock text.
        controls=r.selected_profile()['agent'].get('progressive_config') or {}
        # The graph fallback finds no relation graph in the mock text and stops the same way.
        graph=r.selected_profile()['agent'].get('graph_fallback_v1') is True
        expected='[ESCALATION_REQUIRED]' if controls.get('stop_at_escalation') or graph else 'Offline interface'
        self.assertIn(expected,str(result.get('final_answer') or result.get('final_content')))
        if controls.get('synthesize_at_stop'):
            # The closing call must clear the budget layer and reach the provider.
            self.assertIn('research budget for this question has ended',json.dumps(requests[-1]['messages']))
            saved=next((r.HERE/'native').rglob('convergence_v1.json')).read_text(encoding='utf-8')
            self.assertNotIn('closing_synthesis_failed',saved);self.assertIn('answer_settled',saved)
        r.save(r.HERE/'STATE.json',r.public(result))
        self.assertEqual(meter.held,0)

GRAPH_QUESTION=('A museum that opened in 1950 stands beside a bridge built in the 1930s. '
    'The bridge links two islands. Which of the two islands is larger?')
GRAPH={'nodes':[{'id':'M','kind':'unknown','description':'the museum'},{'id':'B','kind':'unknown','description':'the bridge'},
    {'id':'I','kind':'unknown','description':'the two islands','count':2},{'id':'A','kind':'answer','description':'the larger island'},
    {'id':'Y','kind':'constant','description':'opening year','value':'1950'}],
  'relations':[{'id':'r1','span':'A museum that opened in 1950','subject':'M','relation':'museum opened in','object':'Y','kind':'lookup','depends_on':['Y'],'anchor':True},
    {'id':'r2','span':'stands beside a bridge built in the 1930s','subject':'B','relation':'bridge beside','object':'M','kind':'lookup','depends_on':['M'],'anchor':True},
    {'id':'r3','span':'built in the 1930s','subject':'B','relation':'bridge built in','object':'','kind':'filter','depends_on':[],'anchor':False},
    {'id':'r4','span':'The bridge links two islands','subject':'I','relation':'islands linked by','object':'B','kind':'lookup','depends_on':['B'],'anchor':True},
    {'id':'r5','span':'Which of the two islands is larger?','subject':'A','relation':'larger island of','object':'I','kind':'compare','depends_on':['I'],'anchor':True}],
  'answer':{'node':'A','form':"the island's name"}}
GRAPH_FINDINGS={'museum opened in':('Harbour Museum','The Harbour Museum opened in 1950 by the old port.'),
    'bridge beside':('Old Bridge','The Old Bridge stands beside the Harbour Museum.'),
    'bridge built in':('Old Bridge','The Old Bridge was built in 1934 by the city.'),
    'islands linked by':(['North Isle','South Isle'],'The Old Bridge links North Isle and South Isle across the strait.'),
    'larger island of':('North Isle','North Isle covers 40 square kilometres, more than South Isle.')}

def _text(content):
    return content if isinstance(content,str) else ''.join(str(p.get('text','')) for p in content or [] if isinstance(p,dict))

def _graph_reply(payload):
    """What a scripted model says to one request of the graph path."""
    import re
    messages=payload['messages'];first=_text(next(m['content'] for m in messages if m['role']=='user'))
    def call(name,args):return {'role':'assistant','content':'','tool_calls':[{'id':'call_'+name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
    def say(obj):return {'role':'assistant','content':json.dumps(obj)}
    if payload.get('tools'):
        if first.startswith('Independently re-verify'):
            return call('submit_report',{'content':'Checked. '+json.dumps({'verdict':'confirmed','links':[]}),'confidence':'high'})
        relation=re.search(r'Relation: (.+)',first).group(1).strip()
        if not any(m['role']=='tool' for m in messages):return call('web_search',{'q':relation})
        entity,sentence=GRAPH_FINDINGS[relation];quote=[{'quote':sentence,'url':'https://example.org/'+relation.replace(' ','-')}]
        if 'Candidates to test' in first:found={'verdicts':[{'entity':entity,'satisfies':True,'quotes':quote}]}
        else:found={'candidates':[{'entity':e,'quotes':quote} for e in (entity if isinstance(entity,list) else [entity])]}
        return call('submit_report',{'content':'Findings. '+json.dumps(found),'confidence':'high'})
    prompt=_text(messages[-1]['content']).strip()
    if prompt.startswith(('Decompose','Your relation graph failed','Three independent','The relation graph below','Rewrite the relation graph','Research on this relation graph')):
        return say(GRAPH)
    if prompt.startswith('Below is a relation graph'):
        return say({'conditions':['a museum opened in 1950','it stands beside a bridge built in the 1930s','the bridge links two islands','which of the two islands is larger']})
    if prompt.startswith(('A question and a list','Review this relation graph')):return say({'missing':[],'extra':[],'misread':[],'ambiguous':[]})
    if prompt.startswith('You are checking an answer'):
        return say({'verdict':'supported','relations':[{'relation':i,'established':True} for i in sorted(set(re.findall(r'"relation": "(r\d)"',prompt)))]})
    raise AssertionError('unexpected prompt: '+prompt[:60])

class _Body(httpx.AsyncByteStream):
    """A body the transport must read, as a real one is: a prebuilt Response holds its
    content already, so the metered stream around it would never settle its reservation."""
    def __init__(self,data):self.data=data
    async def __aiter__(self):yield self.data
    async def aclose(self):pass

def _respond(payload,message):
    usage={'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'prompt_cache_hit_tokens':0}
    finish='tool_calls' if message.get('tool_calls') else 'stop'
    if not payload.get('stream'):
        body=json.dumps({'id':'offline','object':'chat.completion','created':0,'model':payload['model'],
            'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':usage}).encode()
        return httpx.Response(200,headers={'content-type':'application/json'},stream=_Body(body))
    delta={'role':'assistant'}
    if message.get('content'):delta['content']=message['content']
    if message.get('tool_calls'):delta['tool_calls']=[dict(c,index=i) for i,c in enumerate(message['tool_calls'])]
    frames=[{'id':'offline','object':'chat.completion.chunk','created':0,'model':payload['model'],'choices':[{'index':0,'delta':delta,'finish_reason':None}]},
        {'id':'offline','object':'chat.completion.chunk','created':0,'model':payload['model'],'choices':[{'index':0,'delta':{},'finish_reason':finish}],'usage':usage}]
    data=b''.join(b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames)+b'data: [DONE]\n\n'
    return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=_Body(data))

async def _graph_fallback_full_path_offline(self):
        """The relation-graph fallback through the real runtime on scripted replies: three
        decompositions, the checks, sub-agents that search and report, the final checks."""
        if r.selected_profile()['agent'].get('graph_fallback_v1') is not True:self.skipTest('graph profile only')
        r.credentials();os.chdir(r.UP)
        meter=r.Meter();r.install_transport(meter)
        from answer_label_guard import install_guard
        install_guard(GRAPH_QUESTION,r.HERE)
        seen={'serper':0,'model':0}
        async def responder(request):
            if request.url.host=='google.serper.dev':
                seen['serper']+=1;q=json.loads(request.content).get('q','')
                sentence=GRAPH_FINDINGS.get(q,('',''))[1]
                return httpx.Response(200,json={'organic':[{'title':'Local history notes','link':'https://example.org/'+q.replace(' ','-'),'snippet':sentence}]})
            if request.url.host!='api.deepseek.com':raise RuntimeError('unexpected host '+request.url.host)
            payload=json.loads(request.content);seen['model']+=1
            if payload.get('tools'):
                first=_text(next(m['content'] for m in payload['messages'] if m['role']=='user'))
                seen.setdefault('reverifier' if first.startswith('Independently re-verify') else 'research',set()).add(payload['model'])
            return _respond(payload,_graph_reply(payload))
        original_init=httpx.AsyncClient.__init__
        def init(client,*a,**kw):
            kw['transport']=httpx.MockTransport(responder);kw['trust_env']=False;original_init(client,*a,**kw)
        with patch.object(httpx.AsyncClient,'__init__',init):
            from benchmarks.public.core.kernel_adapter import BenchmarkSession
            from workflows.agent_team.nodes.main_agent import _llm_cache
            _llm_cache.clear()
            async with BenchmarkSession() as session:
                result=await session.run(GRAPH_QUESTION,pipeline_id='agent_team',
                    meta={'profile':'benchmark','profile_inline':r.selected_profile(),'profile_overrides':r.overrides(),'_trial_dir':str(r.HERE/'native')})
        text=str(result.get('final_answer') or result.get('final_content'))
        state=json.loads(next((r.HERE/'native').rglob('graph_v1.json')).read_text(encoding='utf-8'))
        problems=[e for e in state['events'] if e['event'] in ('subagent_failed','subagent_report_unusable','unsourced_quotes_dropped','model_call_failed','unusable_reply')]
        self.assertEqual(problems,[],state['events'][-6:])
        self.assertIn('North Isle',text);self.assertIn('Evidence level: verified',text)
        self.assertEqual(state['settlement']['level'],'verified')
        self.assertEqual((state['calls']['subagents'],state['calls']['reverifiers']),(10,1))
        self.assertGreaterEqual(seen['serper'],5);self.assertTrue(list((r.HERE/'web_responses').glob('*.json')))
        # Research sub-agents run on the profile's sub-agent model; the re-verifier on the trial's.
        expected=(r.selected_profile()['agent'].get('graph_fallback_config') or {}).get('subagent_model') or os.environ['OPENAI_MODEL']
        self.assertEqual(seen.get('research'),{expected});self.assertEqual(seen.get('reverifier'),{os.environ['OPENAI_MODEL']})
        self.assertEqual(meter.held,0)

FAT_PAGE='https://example.org/old-bridge'
FAT_EVAL_ENTRY='https://huggingface.co/datasets/demo/agent-eval-logs'
# Page reads refuse Hugging Face datasets by themselves, so the page an agent asks for
# is an evaluation source the reader would otherwise fetch.
FAT_EVAL_PAGE='https://www.kaggle.com/datasets/demo/agent-eval-logs'
FAT_SENTENCE='The Old Bridge links North Isle and South Isle. North Isle covers 40 square kilometres, more than South Isle.'
# An ordinary page whose body carries one of the audit's labels.
FAT_LABEL_PAGE='https://example.com/notes'
# A shell download of an evaluation file, which must not run at all.
FAT_SHELL='curl -sL https://huggingface.co/datasets/demo/agent-eval-logs/resolve/main/runs.json'
# Text no model may receive: an evaluation-source entry, the result of an answer-seeking
# search, the body of an evaluation-source page, an ordinary page on an evaluation host,
# a snippet sharing the audit's share of the question, and a labelled page body.
FAT_LEAKS=('LEAKED-SNIPPET-TOKEN','SEEK-RESULT-TOKEN','LEAKED-PAGE-TOKEN','LEAKED-HOST-TOKEN','LEAKED-OVERLAP-TOKEN',
    'LEAKED-LABEL-TOKEN')

def _fat_reply(payload):
    """What a scripted model says to one request of the team path."""
    import re
    messages=payload['messages']
    names=[t.get('function',{}).get('name') for t in payload.get('tools') or []]
    done=[c['function']['name'] for m in messages if m['role']=='assistant' for c in m.get('tool_calls') or []]
    def call(name,args):
        return {'role':'assistant','content':'','tool_calls':[{'id':f'call_{name}_{len(messages)}','type':'function',
            'function':{'name':name,'arguments':json.dumps(args)}}]}
    if 'submit_report' in names:
        steps=[('web_search',{'q':'old bridge islands area'}),('web_search',{'q':'browsecomp reference answer islands'}),
            ('web_fetch',{'url':FAT_PAGE,'info_to_extract':'Which of the two islands is larger?'}),
            ('web_fetch',{'url':FAT_EVAL_PAGE,'info_to_extract':'Which of the two islands is larger?'}),
            ('web_fetch',{'url':FAT_LABEL_PAGE,'info_to_extract':'Which of the two islands is larger?'}),
            ('bash',{'command':FAT_SHELL}),
            ('submit_report',{'content':'Findings: North Isle is the larger island. Evidence: '+FAT_SENTENCE+' URL: '+FAT_PAGE,'confidence':0.9})]
        return call(*steps[min(len(done),len(steps)-1)])
    if names:
        review=next((_text(m['content']) for m in reversed(messages) if m['role']!='assistant'
                     and 'Research reports were received' in _text(m['content'])),'')
        if review and 'review_research' not in done:
            return call('review_research',{'action':'finalize','receipt_ids':sorted(set(re.findall(r'"receipt_id": "([^"]+)"',review))),
                'final_answer':'North Isle','disposition':'The report quotes the island areas.'})
        if 'create_subagent' not in done:
            return call('create_subagent',{'agents':[{'name':'islands_research','system_prompt':'You research islands and bridges.'}]})
        if 'assign_task' not in done:
            return call('assign_task',{'tasks':[{'agent':'islands_research','prompt':'Find which island linked by the Old Bridge is larger. Quote sources.'}]})
        return call('collect_reports',{})
    if 'CONTENT TO ANALYZE' in _text(messages[-1]['content']):
        return {'role':'assistant','content':'North Isle covers 40 square kilometres, more than South Isle.'}
    return {'role':'assistant','content':'North Isle'}

async def _fat_full_path_offline(self):
        """The agent team (FAT) through the real runtime on scripted replies: the coordinator
        creates and assigns a helper, the helper searches, reads pages and reports, and the
        coordinator finalizes through review_research. Evaluation sources and an answer-seeking
        search must be withheld from every model, the page summarizer included."""
        agent=r.selected_profile()['agent']
        if agent.get('convergence_v1') is not True or agent.get('progressive_research_v1') or agent.get('graph_fallback_v1'):
            self.skipTest('team profile only')
        r.credentials();os.chdir(r.UP)
        meter=r.Meter();r.install_transport(meter)
        from answer_label_guard import install_guard
        guard,_=install_guard(GRAPH_QUESTION,r.HERE)
        # The first words of the question, at the audit's 15% share and no more.
        overlap=' '.join(re.findall(r'\w+',GRAPH_QUESTION.casefold())[:guard.width])
        seen={'serper':[],'jina':[],'models':set(),'bodies':[],'hosts':[],'sandboxes':0}
        async def responder(request):
            seen['hosts'].append(str(request.url)[:80])
            if request.url.host=='google.serper.dev':
                q=json.loads(request.content).get('q','');seen['serper'].append(q)
                if 'browsecomp' in q:
                    organic=[{'title':'Benchmark answers','link':'https://example.net/answers','snippet':'SEEK-RESULT-TOKEN North Isle'}]
                else:
                    organic=[{'title':'Old Bridge history','link':FAT_PAGE,'snippet':FAT_SENTENCE},
                        {'title':'agent eval logs · Datasets at Hugging Face','link':FAT_EVAL_ENTRY,'snippet':'LEAKED-SNIPPET-TOKEN North Isle'},
                        {'title':'Island papers','link':'https://huggingface.co/papers/2401.00001','snippet':'LEAKED-HOST-TOKEN islands'},
                        {'title':'Travel notes','link':'https://example.net/travel','snippet':overlap+' LEAKED-OVERLAP-TOKEN'}]
                return httpx.Response(200,json={'organic':organic})
            if request.url.host=='r.jina.ai':
                seen['jina'].append(str(request.url))
                if 'kaggle.com' in str(request.url):body='LEAKED-PAGE-TOKEN The answer is North Isle.'
                elif 'example.com' in str(request.url):body='Island notes. Explanation: LEAKED-LABEL-TOKEN North Isle.'
                else:body='Local history. '+FAT_SENTENCE
                return httpx.Response(200,text=body)
            if request.url.host!='api.deepseek.com':raise RuntimeError('unexpected host '+request.url.host)
            payload=json.loads(request.content);seen['models'].add(payload['model']);seen['bodies'].append(request.content.decode('utf-8','replace'))
            return _respond(payload,_fat_reply(payload))
        original_init=httpx.AsyncClient.__init__
        def init(client,*a,**kw):
            kw['transport']=httpx.MockTransport(responder);kw['trust_env']=False;original_init(client,*a,**kw)
        import importlib
        aligned=importlib.import_module('plugins.tools.web_fetch_aligned')
        bash_tool=importlib.import_module('plugins.tools.bash')
        async def public_url(url):return ''  # the trial has no DNS to vet a host with
        async def no_sandbox():
            seen['sandboxes']+=1;raise RuntimeError('no sandbox in the offline test')
        with patch.object(httpx.AsyncClient,'__init__',init),patch.object(aligned,'non_public_url_error',public_url),\
                patch.object(bash_tool,'aget_sandbox',no_sandbox):
            from benchmarks.public.core.kernel_adapter import BenchmarkSession
            from workflows.agent_team.nodes.main_agent import _llm_cache
            _llm_cache.clear()
            async with BenchmarkSession() as session:
                result=await session.run(GRAPH_QUESTION,pipeline_id='agent_team',
                    meta={'profile':'benchmark','profile_inline':r.selected_profile(),'profile_overrides':r.overrides(),'_trial_dir':str(r.HERE/'native')})
        text=str(result.get('final_answer') or result.get('final_content'))
        state=json.loads(next((r.HERE/'native').rglob('convergence_v1.json')).read_text(encoding='utf-8'))
        self.assertIn('North Isle',text)
        self.assertEqual([d['action'] for d in state['decisions']],['finalize'],state['events'][-6:])
        self.assertTrue(any(x['status']=='complete' and x['agent']!='main' for x in state['receipts']))
        # The helper's two searches and the ordinary page reached the services.
        self.assertEqual(len(seen['serper']),2,seen['serper'])
        self.assertTrue(any('example.org' in u for u in seen['jina']),[h for h in seen['hosts'] if 'deepseek' not in h])
        stages={json.loads(line)['stage'] for line in (r.HERE/'ANSWER_LABEL_GUARD.jsonl').read_text(encoding='utf-8').splitlines()}
        self.assertTrue({'search_eval_source','answer_seeking_request:web_search','fetch_refused_before_read',
            'bash_refused_before_run'}<=stages,stages)
        # The evaluation page is refused before the reader is asked for it, and the shell
        # download before a sandbox is asked for.
        self.assertFalse([u for u in seen['jina'] if 'kaggle.com' in u])
        self.assertEqual(seen['sandboxes'],0)
        self.assertTrue(any('example.com' in u for u in seen['jina']),seen['jina'])
        # Every request a model received, the summarizer's too, is free of withheld text.
        for token in FAT_LEAKS:
            self.assertFalse([b for b in seen['bodies'] if token in b],token)
        # Each model request is saved as sent, for the source audit.
        self.assertEqual(len(list((r.HERE/'requests').glob('*.json'))),len(seen['bodies']))
        self.assertTrue(list((r.HERE/'web_responses').glob('*.json')))
        self.assertEqual(seen['models'],{os.environ['OPENAI_MODEL']})
        self.assertEqual(meter.held,0)

# On Checks, so the trial's standard test command runs it through LinuxChecks.
Checks.test_graph_fallback_full_path_offline=_graph_fallback_full_path_offline
Checks.test_fat_full_path_offline=_fat_full_path_offline

if __name__=='__main__':unittest.main(verbosity=2)
